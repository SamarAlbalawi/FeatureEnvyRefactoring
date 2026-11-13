using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.MSBuild;
using Microsoft.CodeAnalysis.Text;

namespace CSharpDependencyAnalyzer
{
    internal static class AffectedScopeAnalyzer
    {
        internal static async Task<int> RunAsync(string[] args)
        {
            try
            {
                var options = ScopeOptions.Parse(args);
                if (!MSBuildLocator.IsRegistered)
                {
                    var msbuildPath = Environment.GetEnvironmentVariable("ROSLYN_MSBUILD_PATH");
                    if (string.IsNullOrWhiteSpace(msbuildPath))
                    {
                        throw new InvalidOperationException(
                            "Set ROSLYN_MSBUILD_PATH to the local MSBuild SDK directory.");
                    }

                    MSBuildLocator.RegisterMSBuildPath(msbuildPath);
                }

                var workspaceDiagnostics = new List<string>();
                using (var workspace = MSBuildWorkspace.Create())
                {
                    workspace.WorkspaceFailed += (_, eventArgs) =>
                        workspaceDiagnostics.Add(eventArgs.Diagnostic.Kind + ": " + eventArgs.Diagnostic.Message);

                    var project = await workspace.OpenProjectAsync(
                        Path.GetFullPath(options.ProjectPath),
                        cancellationToken: CancellationToken.None);
                    var document = project.Documents.FirstOrDefault(candidate =>
                        string.Equals(
                            Path.GetFullPath(candidate.FilePath ?? string.Empty),
                            Path.GetFullPath(options.SourceFilePath),
                            StringComparison.OrdinalIgnoreCase));
                    if (document == null)
                    {
                        throw new InvalidOperationException(
                            "The source file is not included in the requested Roslyn project: " +
                            Path.GetFullPath(options.SourceFilePath));
                    }

                    var root = await document.GetSyntaxRootAsync();
                    var semanticModel = await document.GetSemanticModelAsync();
                    var compilation = await project.GetCompilationAsync();
                    if (root == null || semanticModel == null || compilation == null)
                    {
                        throw new InvalidOperationException(
                            "Roslyn did not produce the project compilation and target semantic model.");
                    }

                    var targetTree = document.GetSyntaxTreeAsync().GetAwaiter().GetResult();
                    if (targetTree == null)
                    {
                        throw new InvalidOperationException("Roslyn did not produce the target syntax tree.");
                    }

                    var after = AnalyzeSnapshot(compilation, targetTree, options);
                    ScopeSnapshot? before = null;
                    var baselineChanges = new List<BaselineTreeChange>();
                    if (!string.IsNullOrWhiteSpace(options.BaselineRootPath))
                    {
                        var patchedRoot = Path.GetFullPath(options.PatchedRootPath ?? string.Empty);
                        var baselineRoot = Path.GetFullPath(options.BaselineRootPath);
                        var beforeCompilation = compilation;
                        foreach (var syntaxTree in compilation.SyntaxTrees.ToList())
                        {
                            var treePath = Path.GetFullPath(syntaxTree.FilePath ?? string.Empty);
                            var relativePath = Path.GetRelativePath(patchedRoot, treePath);
                            if (Path.IsPathRooted(relativePath) ||
                                relativePath == ".." ||
                                relativePath.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                            {
                                continue;
                            }

                            var baselinePath = Path.GetFullPath(Path.Combine(baselineRoot, relativePath));
                            if (!File.Exists(baselinePath))
                            {
                                beforeCompilation = beforeCompilation.RemoveSyntaxTrees(syntaxTree);
                                baselineChanges.Add(new BaselineTreeChange
                                {
                                    Change = "REMOVED_ADDED_TREE",
                                    PatchedFile = treePath,
                                    BaselineFile = baselinePath
                                });
                                continue;
                            }

                            var baselineContent = File.ReadAllText(baselinePath);
                            var baselineTree = CSharpSyntaxTree.ParseText(
                                SourceText.From(baselineContent),
                                syntaxTree.Options as CSharpParseOptions,
                                syntaxTree.FilePath ?? string.Empty);
                            beforeCompilation = beforeCompilation.ReplaceSyntaxTree(syntaxTree, baselineTree);
                            baselineChanges.Add(new BaselineTreeChange
                            {
                                Change = string.Equals(
                                    baselineContent,
                                    syntaxTree.GetText().ToString(),
                                    StringComparison.Ordinal)
                                    ? "UNCHANGED_TREE"
                                    : "REPLACED_WITH_PINNED_SOURCE",
                                PatchedFile = treePath,
                                BaselineFile = baselinePath
                            });
                        }

                        var beforeTargetTree = beforeCompilation.SyntaxTrees.FirstOrDefault(tree =>
                            string.Equals(
                                Path.GetFullPath(tree.FilePath ?? string.Empty),
                                Path.GetFullPath(targetTree.FilePath ?? string.Empty),
                                StringComparison.OrdinalIgnoreCase));
                        if (beforeTargetTree == null)
                        {
                            throw new InvalidOperationException(
                                "The target file is absent from the pinned baseline project compilation.");
                        }

                        before = AnalyzeSnapshot(beforeCompilation, beforeTargetTree, options);
                    }
                    else if (!string.IsNullOrWhiteSpace(options.BeforeSourceFilePath))
                    {
                        var beforeContent = File.ReadAllText(options.BeforeSourceFilePath);
                        var beforeTree = CSharpSyntaxTree.ParseText(
                            SourceText.From(beforeContent),
                            targetTree.Options as CSharpParseOptions,
                            targetTree.FilePath);
                        var beforeCompilation = compilation.ReplaceSyntaxTree(targetTree, beforeTree);
                        if (!string.IsNullOrWhiteSpace(options.RemovedFilePath))
                        {
                            var removedPath = Path.GetFullPath(options.RemovedFilePath);
                            var removedTree = beforeCompilation.SyntaxTrees.FirstOrDefault(tree =>
                                string.Equals(
                                    Path.GetFullPath(tree.FilePath ?? string.Empty),
                                    removedPath,
                                    StringComparison.OrdinalIgnoreCase));
                            if (removedTree == null)
                            {
                                throw new InvalidOperationException(
                                    "The declared added file is not part of the patched project: " + removedPath);
                            }

                            beforeCompilation = beforeCompilation.RemoveSyntaxTrees(removedTree);
                        }

                        before = AnalyzeSnapshot(beforeCompilation, beforeTree, options);
                    }

                    var result = new ScopeAnalysisResult
                    {
                        SchemaVersion = "1.0",
                        Repository = new ScopeRepositoryInfo
                        {
                            ProjectIdentifier = project.Name,
                            ProjectPath = Path.GetFullPath(options.ProjectPath),
                            TargetFile = Path.GetFullPath(document.FilePath ?? options.SourceFilePath)
                        },
                        Before = before,
                        BaselineTreeChanges = baselineChanges,
                        After = after,
                        Target = after.Target,
                        TargetDependencies = after.TargetDependencies,
                        DirectCallers = after.DirectCallers,
                        ResolvedSameNameCalls = after.ResolvedSameNameCalls,
                        CompilerDiagnostics = after.CompilerDiagnostics,
                        WorkspaceDiagnostics = workspaceDiagnostics
                    };

                    var json = JsonSerializer.Serialize(
                        result,
                        new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(options.OutputPath, json);
                }

                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception);
                return 1;
            }
        }

        private static ScopeSnapshot AnalyzeSnapshot(
            Compilation compilation,
            SyntaxTree targetTree,
            ScopeOptions options)
        {
            var root = targetTree.GetRoot();
            var semanticModel = compilation.GetSemanticModel(targetTree);
            var targetMethod = FindTargetMethod(root, semanticModel, options);
            var targetSymbol = semanticModel.GetDeclaredSymbol(targetMethod) as IMethodSymbol;
            if (targetSymbol == null)
            {
                throw new InvalidOperationException("Roslyn could not resolve the target method symbol.");
            }

            var collector = new DependencyCollector(semanticModel, targetMethod, targetSymbol.ContainingType);
            collector.Collect();

            var callers = new List<CallSiteInfo>();
            var sameNameCalls = new List<CallSiteInfo>();
            foreach (var syntaxTree in compilation.SyntaxTrees)
            {
                var treeRoot = syntaxTree.GetRoot();
                var model = compilation.GetSemanticModel(syntaxTree);
                foreach (var invocation in treeRoot.DescendantNodes().OfType<InvocationExpressionSyntax>())
                {
                    var invoked = model.GetSymbolInfo(invocation).Symbol as IMethodSymbol;
                    if (invoked == null ||
                        !string.Equals(invoked.Name, targetSymbol.Name, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    var site = CreateCallSite(invocation, model, invoked);
                    sameNameCalls.Add(site);
                    if (SymbolEqualityComparer.Default.Equals(
                        invoked.OriginalDefinition,
                        targetSymbol.OriginalDefinition))
                    {
                        callers.Add(site);
                    }
                }
            }

            var diagnostics = compilation.GetDiagnostics()
                .Select(DiagnosticInfo.Create)
                .OrderBy(item => item.File, StringComparer.OrdinalIgnoreCase)
                .ThenBy(item => item.StartLine)
                .ThenBy(item => item.Id, StringComparer.Ordinal)
                .ToList();

            return new ScopeSnapshot
            {
                Target = new ScopeTargetInfo
                {
                    ContainingType = targetSymbol.ContainingType.ToDisplayString(),
                    MethodName = targetSymbol.Name,
                    Signature = targetSymbol.ToDisplayString(),
                    Accessibility = targetSymbol.DeclaredAccessibility.ToString(),
                    IsStatic = targetSymbol.IsStatic
                },
                TargetDependencies = new ScopeDependencyInfo
                {
                    Callees = collector.Callees,
                    Fields = collector.Fields,
                    Properties = collector.Properties,
                    ReferencedTypes = collector.ReferencedTypes,
                    LocalDependencies = collector.LocalDependencies,
                    ForeignDependencies = collector.ForeignDependencies,
                    UnresolvedDependencies = collector.Unresolved
                },
                DirectCallers = callers,
                ResolvedSameNameCalls = sameNameCalls,
                CompilerDiagnostics = diagnostics
            };
        }

        private static SyntaxNode FindTargetMethod(
            SyntaxNode root,
            SemanticModel semanticModel,
            ScopeOptions options)
        {
            var matches = root.DescendantNodes()
                .Where(node =>
                    (node is MethodDeclarationSyntax method &&
                     method.Identifier.ValueText == options.MethodName) ||
                    (node is ConstructorDeclarationSyntax constructor &&
                     constructor.Identifier.ValueText == options.MethodName))
                .Where(node =>
                {
                    var symbol = semanticModel.GetDeclaredSymbol(node) as IMethodSymbol;
                    return symbol != null &&
                        (string.IsNullOrWhiteSpace(options.ContainingType) ||
                         string.Equals(
                             symbol.ContainingType.ToDisplayString(),
                             options.ContainingType,
                             StringComparison.Ordinal) ||
                         string.Equals(
                             symbol.ContainingType.Name,
                             options.ContainingType,
                             StringComparison.Ordinal));
                })
                .ToList();

            if (!string.IsNullOrWhiteSpace(options.Signature))
            {
                var signatureMatches = matches.Where(node =>
                    string.Equals(
                        semanticModel.GetDeclaredSymbol(node)?.ToDisplayString(),
                        options.Signature,
                        StringComparison.Ordinal)).ToList();
                if (signatureMatches.Count > 0)
                {
                    matches = signatureMatches;
                }
            }

            if (matches.Count > 1 && options.SourceLine.HasValue)
            {
                matches = matches.Where(node =>
                    node.GetLocation().GetLineSpan().StartLinePosition.Line + 1 ==
                    options.SourceLine.Value).ToList();
            }

            if (matches.Count != 1)
            {
                throw new InvalidOperationException(
                    "Expected exactly one target method, found " + matches.Count + ".");
            }

            return matches[0];
        }

        private static CallSiteInfo CreateCallSite(
            InvocationExpressionSyntax invocation,
            SemanticModel semanticModel,
            IMethodSymbol invoked)
        {
            var location = invocation.GetLocation().GetLineSpan();
            var caller = semanticModel.GetEnclosingSymbol(invocation.SpanStart) as IMethodSymbol;
            return new CallSiteInfo
            {
                InvokedSymbol = invoked.ToDisplayString(),
                InvokedContainingType = invoked.ContainingType?.ToDisplayString() ?? string.Empty,
                CallerSymbol = caller?.ToDisplayString() ?? string.Empty,
                CallerContainingType = caller?.ContainingType.ToDisplayString() ?? string.Empty,
                Invocation = invocation.ToString(),
                File = invocation.SyntaxTree.FilePath ?? string.Empty,
                StartLine = location.StartLinePosition.Line + 1,
                StartColumn = location.StartLinePosition.Character + 1
            };
        }
    }

    internal sealed class ScopeOptions
    {
        internal string ProjectPath { get; private set; } = string.Empty;
        internal string SourceFilePath { get; private set; } = string.Empty;
        internal string ContainingType { get; private set; } = string.Empty;
        internal string MethodName { get; private set; } = string.Empty;
        internal string? Signature { get; private set; }
        internal int? SourceLine { get; private set; }
        internal string OutputPath { get; private set; } = string.Empty;
        internal string? BeforeSourceFilePath { get; private set; }
        internal string? RemovedFilePath { get; private set; }
        internal string? BaselineRootPath { get; private set; }
        internal string? PatchedRootPath { get; private set; }

        internal static ScopeOptions Parse(string[] args)
        {
            var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var index = 0; index < args.Length; index++)
            {
                if (!args[index].StartsWith("--", StringComparison.Ordinal) || index + 1 >= args.Length)
                {
                    throw new ArgumentException("Arguments must use --name value form.");
                }

                values[args[index].Substring(2)] = args[++index];
            }

            var options = new ScopeOptions
            {
                ProjectPath = Required(values, "project"),
                SourceFilePath = Required(values, "source-file"),
                MethodName = Required(values, "method"),
                OutputPath = Required(values, "output"),
                ContainingType = values.TryGetValue("containing-type", out var containingType)
                    ? containingType
                    : string.Empty,
                Signature = values.TryGetValue("signature", out var signature) ? signature : null,
                SourceLine = values.TryGetValue("source-line", out var sourceLine) &&
                    int.TryParse(sourceLine, out var parsedSourceLine)
                    ? parsedSourceLine
                    : null,
                BeforeSourceFilePath = values.TryGetValue("before-source-file", out var beforeSourceFile)
                    ? Path.GetFullPath(beforeSourceFile)
                    : null,
                RemovedFilePath = values.TryGetValue("removed-file", out var removedFile)
                    ? Path.GetFullPath(removedFile)
                    : null,
                BaselineRootPath = values.TryGetValue("baseline-root", out var baselineRoot)
                    ? Path.GetFullPath(baselineRoot)
                    : null,
                PatchedRootPath = values.TryGetValue("patched-root", out var patchedRoot)
                    ? Path.GetFullPath(patchedRoot)
                    : null
            };
            if (options.BaselineRootPath != null && options.PatchedRootPath == null)
            {
                throw new ArgumentException("--baseline-root requires --patched-root.");
            }

            return options;
        }

        private static string Required(Dictionary<string, string> values, string name)
        {
            if (!values.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException("Missing required argument --" + name + ".");
            }

            return value;
        }
    }

    internal sealed class ScopeAnalysisResult
    {
        public string SchemaVersion { get; set; } = string.Empty;
        public ScopeRepositoryInfo Repository { get; set; } = new ScopeRepositoryInfo();
        public ScopeSnapshot? Before { get; set; }
        public List<BaselineTreeChange> BaselineTreeChanges { get; set; } = new List<BaselineTreeChange>();
        public ScopeSnapshot After { get; set; } = new ScopeSnapshot();
        public ScopeTargetInfo Target { get; set; } = new ScopeTargetInfo();
        public ScopeDependencyInfo TargetDependencies { get; set; } = new ScopeDependencyInfo();
        public List<CallSiteInfo> DirectCallers { get; set; } = new List<CallSiteInfo>();
        public List<CallSiteInfo> ResolvedSameNameCalls { get; set; } = new List<CallSiteInfo>();
        public List<DiagnosticInfo> CompilerDiagnostics { get; set; } = new List<DiagnosticInfo>();
        public List<string> WorkspaceDiagnostics { get; set; } = new List<string>();
    }

    internal sealed class ScopeSnapshot
    {
        public ScopeTargetInfo Target { get; set; } = new ScopeTargetInfo();
        public ScopeDependencyInfo TargetDependencies { get; set; } = new ScopeDependencyInfo();
        public List<CallSiteInfo> DirectCallers { get; set; } = new List<CallSiteInfo>();
        public List<CallSiteInfo> ResolvedSameNameCalls { get; set; } = new List<CallSiteInfo>();
        public List<DiagnosticInfo> CompilerDiagnostics { get; set; } = new List<DiagnosticInfo>();
    }

    internal sealed class BaselineTreeChange
    {
        public string Change { get; set; } = string.Empty;
        public string PatchedFile { get; set; } = string.Empty;
        public string BaselineFile { get; set; } = string.Empty;
    }

    internal sealed class ScopeRepositoryInfo
    {
        public string ProjectIdentifier { get; set; } = string.Empty;
        public string ProjectPath { get; set; } = string.Empty;
        public string TargetFile { get; set; } = string.Empty;
    }

    internal sealed class ScopeTargetInfo
    {
        public string ContainingType { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public string Accessibility { get; set; } = string.Empty;
        public bool IsStatic { get; set; }
    }

    internal sealed class ScopeDependencyInfo
    {
        public List<DependencyReference> Callees { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> Fields { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> Properties { get; set; } = new List<DependencyReference>();
        public List<ReferenceInfo> ReferencedTypes { get; set; } = new List<ReferenceInfo>();
        public List<DependencyReference> LocalDependencies { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> ForeignDependencies { get; set; } = new List<DependencyReference>();
        public List<UnresolvedDependency> UnresolvedDependencies { get; set; } = new List<UnresolvedDependency>();
    }

    internal sealed class CallSiteInfo
    {
        public string InvokedSymbol { get; set; } = string.Empty;
        public string InvokedContainingType { get; set; } = string.Empty;
        public string CallerSymbol { get; set; } = string.Empty;
        public string CallerContainingType { get; set; } = string.Empty;
        public string Invocation { get; set; } = string.Empty;
        public string File { get; set; } = string.Empty;
        public int StartLine { get; set; }
        public int StartColumn { get; set; }
    }

    internal sealed class DiagnosticInfo
    {
        public string Id { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string File { get; set; } = string.Empty;
        public int StartLine { get; set; }
        public int StartColumn { get; set; }

        internal static DiagnosticInfo Create(Diagnostic diagnostic)
        {
            var line = diagnostic.Location.GetLineSpan();
            return new DiagnosticInfo
            {
                Id = diagnostic.Id,
                Severity = diagnostic.Severity.ToString(),
                Message = diagnostic.GetMessage(),
                File = diagnostic.Location.IsInSource ? line.Path : string.Empty,
                StartLine = diagnostic.Location.IsInSource
                    ? line.StartLinePosition.Line + 1
                    : 0,
                StartColumn = diagnostic.Location.IsInSource
                    ? line.StartLinePosition.Character + 1
                    : 0
            };
        }
    }
}
