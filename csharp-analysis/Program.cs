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

namespace CSharpDependencyAnalyzer
{
    internal static class Program
    {
        private static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 &&
                string.Equals(args[0], "--mode", StringComparison.OrdinalIgnoreCase) &&
                args.Length > 1 &&
                string.Equals(args[1], "corpus", StringComparison.OrdinalIgnoreCase))
            {
                return await CorpusBuilder.RunAsync(args.Skip(2).ToArray());
            }
            if (args.Length > 0 &&
                string.Equals(args[0], "--mode", StringComparison.OrdinalIgnoreCase) &&
                args.Length > 1 &&
                string.Equals(args[1], "dependencies", StringComparison.OrdinalIgnoreCase))
            {
                return await CorpusBuilder.RunDependenciesAsync(args.Skip(2).ToArray());
            }
            if (args.Length > 0 &&
                string.Equals(args[0], "--mode", StringComparison.OrdinalIgnoreCase) &&
                args.Length > 1 &&
                string.Equals(args[1], "bm25", StringComparison.OrdinalIgnoreCase))
            {
                return Bm25Retriever.Run(args.Skip(2).ToArray());
            }
            if (args.Length > 0 &&
                string.Equals(args[0], "--mode", StringComparison.OrdinalIgnoreCase) &&
                args.Length > 1 &&
                string.Equals(args[1], "affected-scope", StringComparison.OrdinalIgnoreCase))
            {
                return await AffectedScopeAnalyzer.RunAsync(args.Skip(2).ToArray());
            }

            try
            {
                var options = Options.Parse(args);
                var result = await AnalyzeAsync(options);
                var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
                var json = JsonSerializer.Serialize(result, jsonOptions);

                if (options.OutputPath == null)
                {
                    Console.WriteLine(json);
                }
                else
                {
                    File.WriteAllText(options.OutputPath, json);
                }

                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception.Message);
                return 1;
            }
        }

        private static async Task<AnalysisResult> AnalyzeAsync(Options options)
        {
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

            var diagnostics = new List<string>();
            using (var workspace = MSBuildWorkspace.Create())
            {
                workspace.WorkspaceFailed += (_, eventArgs) =>
                    diagnostics.Add(eventArgs.Diagnostic.Kind + ": " + eventArgs.Diagnostic.Message);

                var project = await workspace.OpenProjectAsync(
                    Path.GetFullPath(options.ProjectPath),
                    cancellationToken: CancellationToken.None);
                var document = FindDocument(project, options.SourceFilePath);
                var root = await document.GetSyntaxRootAsync();
                var semanticModel = await document.GetSemanticModelAsync();

                if (root == null || semanticModel == null)
                {
                    throw new InvalidOperationException("Roslyn did not produce the target document semantic model.");
                }

                var method = FindMethod(root, semanticModel, options);
                var methodSymbol = semanticModel.GetDeclaredSymbol(method);
                if (methodSymbol == null)
                {
                    throw new InvalidOperationException("Roslyn could not resolve the target method symbol.");
                }

                var containingType = methodSymbol.ContainingType;
                var collector = new DependencyCollector(semanticModel, method, containingType);
                collector.Collect();

                return new AnalysisResult
                {
                    SchemaVersion = "1.0",
                    Repository = new RepositoryInfo
                    {
                        RepositoryIdentifier = options.RepositoryIdentifier,
                        Commit = options.Commit,
                        ProjectPath = Path.GetFullPath(options.ProjectPath),
                        ProjectIdentifier = project.Name,
                        SourceFile = Path.GetFullPath(document.FilePath ?? options.SourceFilePath)
                    },
                    Target = new TargetInfo
                    {
                        ContainingType = containingType.ToDisplayString(),
                        MethodName = methodSymbol.Name,
                        Signature = methodSymbol.ToDisplayString(),
                        DatasetAnnotation = options.DatasetAnnotation
                    },
                    Callees = collector.Callees,
                    Fields = collector.Fields,
                    Properties = collector.Properties,
                    ReferencedTypes = collector.ReferencedTypes,
                    Inheritance = CreateTypeReference(containingType.BaseType, "INHERITANCE"),
                    Interfaces = containingType.Interfaces
                        .Select(type => CreateTypeReference(type, "INTERFACE"))
                        .ToList(),
                    LocalDependencies = collector.LocalDependencies,
                    ForeignDependencies = collector.ForeignDependencies,
                    ForeignDependenciesByDeclaringType = collector.GroupForeignDependencies(),
                    LikelyEnviedTypeCandidates = collector.RankEnviedTypes(),
                    DependencyNeighborhood = collector.Neighborhood,
                    UnresolvedDependencies = collector.Unresolved,
                    WorkspaceDiagnostics = diagnostics
                };
            }
        }

        private static Document FindDocument(Project project, string sourceFilePath)
        {
            var requestedPath = Path.GetFullPath(sourceFilePath);
            var document = project.Documents.FirstOrDefault(candidate =>
                string.Equals(
                    Path.GetFullPath(candidate.FilePath ?? string.Empty),
                    requestedPath,
                    StringComparison.OrdinalIgnoreCase));

            return document ?? throw new InvalidOperationException(
                "The source file is not included in the requested Roslyn project: " + requestedPath);
        }

        private static SyntaxNode FindMethod(
            SyntaxNode root,
            SemanticModel semanticModel,
            Options options)
        {
            var methods = root.DescendantNodes()
                .Where(node => (node is MethodDeclarationSyntax method &&
                                method.Identifier.ValueText == options.MethodName) ||
                               (node is ConstructorDeclarationSyntax constructor &&
                                constructor.Identifier.ValueText == options.MethodName))
                .Where(method =>
                {
                    var symbol = semanticModel.GetDeclaredSymbol(method);
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
                var signatureMatches = methods.Where(method =>
                    string.Equals(
                        GetComparableSignature(
                            semanticModel.GetDeclaredSymbol(method) as IMethodSymbol),
                        GetComparableSignature(options.Signature),
                        StringComparison.Ordinal)).ToList();
                if (signatureMatches.Count > 0)
                {
                    methods = signatureMatches;
                }
            }

            if (methods.Count > 1 && options.SourceLine.HasValue)
            {
                methods = methods.Where(method =>
                    method.GetLocation().GetLineSpan().StartLinePosition.Line + 1 ==
                    options.SourceLine.Value).ToList();
            }

            if (methods.Count != 1)
            {
                throw new InvalidOperationException(
                    "Expected exactly one target method, found " + methods.Count + ".");
            }

            return methods[0];
        }

        private static string GetComparableSignature(IMethodSymbol? symbol)
        {
            return symbol == null ? string.Empty : GetComparableSignature(symbol.ToDisplayString());
        }

        private static string GetComparableSignature(string signature)
        {
            return new string(signature.Where(character => !char.IsWhiteSpace(character)).ToArray());
        }

        private static ReferenceInfo CreateTypeReference(ITypeSymbol? symbol, string relationship)
        {
            if (symbol == null)
            {
                return new ReferenceInfo { Relationship = relationship, ResolutionStatus = "UNRESOLVED" };
            }

            return new ReferenceInfo
            {
                Relationship = relationship,
                ResolutionStatus = "RESOLVED",
                DeclaringType = symbol.ToDisplayString(),
                Member = symbol.Name,
                Location = LocationInfo.Create(symbol)
            };
        }
    }

    internal sealed class DependencyCollector
    {
        private readonly SemanticModel _semanticModel;
        private readonly SyntaxNode _method;
        private readonly INamedTypeSymbol _containingType;
        private readonly Dictionary<string, DependencySummary> _foreignByType =
            new Dictionary<string, DependencySummary>(StringComparer.Ordinal);

        internal DependencyCollector(
            SemanticModel semanticModel,
            SyntaxNode method,
            INamedTypeSymbol containingType)
        {
            _semanticModel = semanticModel;
            _method = method;
            _containingType = containingType;
        }

        internal List<DependencyReference> Callees { get; } = new List<DependencyReference>();
        internal List<DependencyReference> Fields { get; } = new List<DependencyReference>();
        internal List<DependencyReference> Properties { get; } = new List<DependencyReference>();
        internal List<ReferenceInfo> ReferencedTypes { get; } = new List<ReferenceInfo>();
        internal List<DependencyReference> LocalDependencies { get; } = new List<DependencyReference>();
        internal List<DependencyReference> ForeignDependencies { get; } = new List<DependencyReference>();
        internal List<DependencyReference> Neighborhood { get; } = new List<DependencyReference>();
        internal List<UnresolvedDependency> Unresolved { get; } = new List<UnresolvedDependency>();

        internal void Collect()
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);

            foreach (var invocation in _method.DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                var symbol = _semanticModel.GetSymbolInfo(invocation).Symbol as IMethodSymbol;
                if (symbol == null)
                {
                    AddUnresolved("CALL", invocation.ToString());
                    continue;
                }

                AddReference(Callees, symbol, "CALL", seen);
            }

            foreach (var memberAccess in _method.DescendantNodes().OfType<MemberAccessExpressionSyntax>())
            {
                if (memberAccess.Parent is InvocationExpressionSyntax invocation &&
                    invocation.Expression == memberAccess)
                {
                    continue;
                }

                var symbol = _semanticModel.GetSymbolInfo(memberAccess).Symbol;
                if (symbol is IFieldSymbol field)
                {
                    AddReference(Fields, field, "FIELD_ACCESS", seen);
                }
                else if (symbol is IPropertySymbol property)
                {
                    AddReference(Properties, property, "PROPERTY_ACCESS", seen);
                }
                else if (symbol == null)
                {
                    AddUnresolved("MEMBER_ACCESS", memberAccess.ToString());
                }
            }

            foreach (var identifier in _method.DescendantNodes().OfType<IdentifierNameSyntax>())
            {
                if (identifier.Parent is MemberAccessExpressionSyntax memberAccess &&
                    memberAccess.Name == identifier)
                {
                    continue;
                }

                var symbol = _semanticModel.GetSymbolInfo(identifier).Symbol;
                if (symbol is IFieldSymbol field)
                {
                    AddReference(Fields, field, "FIELD_ACCESS", seen);
                }
                else if (symbol is IPropertySymbol property)
                {
                    AddReference(Properties, property, "PROPERTY_ACCESS", seen);
                }
            }

            foreach (var typeSyntax in _method.DescendantNodes()
                .OfType<TypeSyntax>()
                .Where(IsExplicitTypeSyntax))
            {
                var type = _semanticModel.GetTypeInfo(typeSyntax).Type;
                type ??= _semanticModel.GetSymbolInfo(typeSyntax).Symbol as ITypeSymbol;
                if (type is INamedTypeSymbol namedType)
                {
                    AddTypeReference(namedType);
                }
                else if (type == null)
                {
                    AddUnresolved("TYPE_REFERENCE", typeSyntax.ToString());
                }
            }

        }

            private static bool IsExplicitTypeSyntax(TypeSyntax typeSyntax)
            {
                if (typeSyntax.Parent is VariableDeclarationSyntax variableDeclaration &&
                    variableDeclaration.Type == typeSyntax)
                {
                    return true;
                }

                if (typeSyntax.Parent is ObjectCreationExpressionSyntax objectCreation &&
                    objectCreation.Type == typeSyntax)
                {
                    return true;
                }

                if (typeSyntax.Parent is TypeOfExpressionSyntax typeOfExpression &&
                    typeOfExpression.Type == typeSyntax)
                {
                    return true;
                }

                if (typeSyntax.Parent is CastExpressionSyntax castExpression &&
                    castExpression.Type == typeSyntax)
                {
                    return true;
                }

                if (typeSyntax.Parent is DefaultExpressionSyntax defaultExpression &&
                    defaultExpression.Type == typeSyntax)
                {
                    return true;
                }

                return false;
            }

        internal Dictionary<string, DependencySummary> GroupForeignDependencies()
        {
            return _foreignByType
                .OrderByDescending(pair => pair.Value.TotalReferences)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        }

        internal List<EnviedTypeCandidate> RankEnviedTypes()
        {
            return _foreignByType
                .Where(pair => pair.Value.IsSourceType)
                .OrderByDescending(pair => pair.Value.TotalReferences)
                .ThenBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => new EnviedTypeCandidate
                {
                    DeclaringType = pair.Key,
                    Score = pair.Value.TotalReferences,
                    FieldAccesses = pair.Value.FieldAccesses,
                    PropertyAccesses = pair.Value.PropertyAccesses,
                    MethodCalls = pair.Value.MethodCalls
                })
                .ToList();
        }

        private void AddReference(
            List<DependencyReference> destination,
            ISymbol symbol,
            string relationship,
            HashSet<string> seen)
        {
            var key = relationship + "|" + symbol.ToDisplayString() + "|" +
                GetSourceSpan(symbol);
            if (!seen.Add(key))
            {
                return;
            }

            var reference = new DependencyReference
            {
                Relationship = relationship,
                Member = symbol.Name,
                DeclaringType = symbol.ContainingType?.ToDisplayString() ?? symbol.ContainingNamespace.ToDisplayString(),
                Symbol = symbol.ToDisplayString(),
                Scope = IsLocal(symbol) ? "LOCAL" : "FOREIGN",
                Location = LocationInfo.Create(symbol)
            };
            destination.Add(reference);
            Neighborhood.Add(reference);

            if (reference.Scope == "LOCAL")
            {
                LocalDependencies.Add(reference);
            }
            else
            {
                ForeignDependencies.Add(reference);
                AddForeignSummary(reference);
            }
        }

        private void AddTypeReference(INamedTypeSymbol type)
        {
            if (ReferencedTypes.Any(existing => existing.DeclaringType == type.ToDisplayString()))
            {
                return;
            }

            var reference = new ReferenceInfo
            {
                Relationship = "TYPE_REFERENCE",
                ResolutionStatus = "RESOLVED",
                DeclaringType = type.ToDisplayString(),
                Member = type.Name,
                Location = LocationInfo.Create(type)
            };
            ReferencedTypes.Add(reference);
            Neighborhood.Add(new DependencyReference
            {
                Relationship = "TYPE_REFERENCE",
                Member = type.Name,
                DeclaringType = type.ToDisplayString(),
                Symbol = type.ToDisplayString(),
                Scope = IsLocal(type) ? "LOCAL" : "FOREIGN",
                Location = reference.Location
            });
        }

        private void AddForeignSummary(DependencyReference reference)
        {
            if (!_foreignByType.TryGetValue(reference.DeclaringType, out var summary))
            {
                summary = new DependencySummary
                {
                    DeclaringType = reference.DeclaringType,
                    IsSourceType = reference.Location?.IsSource == true
                };
                _foreignByType.Add(reference.DeclaringType, summary);
            }

            summary.TotalReferences++;
            if (reference.Relationship == "FIELD_ACCESS")
            {
                summary.FieldAccesses++;
            }
            else if (reference.Relationship == "PROPERTY_ACCESS")
            {
                summary.PropertyAccesses++;
            }
            else if (reference.Relationship == "CALL")
            {
                summary.MethodCalls++;
            }
        }

        private void AddUnresolved(string relationship, string expression)
        {
            Unresolved.Add(new UnresolvedDependency
            {
                Relationship = relationship,
                Expression = expression,
                Reason = "Roslyn returned no semantic symbol."
            });
        }

        private bool IsLocal(ISymbol symbol)
        {
            return symbol.ContainingType != null &&
                SymbolEqualityComparer.Default.Equals(symbol.ContainingType, _containingType);
        }

        private static string GetSourceSpan(ISymbol symbol)
        {
            var location = symbol.Locations.FirstOrDefault(candidate => candidate.IsInSource);
            return location == null ? string.Empty : location.SourceSpan.ToString();
        }
    }

    internal sealed class Options
    {
        internal string ProjectPath { get; private set; } = string.Empty;
        internal string SourceFilePath { get; private set; } = string.Empty;
        internal string ContainingType { get; private set; } = string.Empty;
        internal string MethodName { get; private set; } = string.Empty;
        internal string? Signature { get; private set; }
        internal int? SourceLine { get; private set; }
        internal string? OutputPath { get; private set; }
        internal string RepositoryIdentifier { get; private set; } = "unknown";
        internal string Commit { get; private set; } = "unknown";
        internal string DatasetAnnotation { get; private set; } = "unknown";

        internal static Options Parse(string[] args)
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

            var options = new Options
            {
                ProjectPath = Required(values, "project"),
                SourceFilePath = Required(values, "source-file"),
                ContainingType = values.TryGetValue("containing-type", out var containingType) ? containingType : string.Empty,
                MethodName = Required(values, "method"),
                Signature = values.TryGetValue("signature", out var signature) ? signature : null,
                SourceLine = values.TryGetValue("source-line", out var sourceLine) &&
                    int.TryParse(sourceLine, out var parsedSourceLine)
                    ? parsedSourceLine
                    : null,
                OutputPath = values.TryGetValue("output", out var output) ? output : null,
                RepositoryIdentifier = values.TryGetValue("repository", out var repository) ? repository : "unknown",
                Commit = values.TryGetValue("commit", out var commit) ? commit : "unknown",
                DatasetAnnotation = values.TryGetValue("dataset-annotation", out var annotation) ? annotation : "unknown"
            };
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

    internal sealed class AnalysisResult
    {
        public string SchemaVersion { get; set; } = string.Empty;
        public RepositoryInfo Repository { get; set; } = new RepositoryInfo();
        public TargetInfo Target { get; set; } = new TargetInfo();
        public List<DependencyReference> Callees { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> Fields { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> Properties { get; set; } = new List<DependencyReference>();
        public List<ReferenceInfo> ReferencedTypes { get; set; } = new List<ReferenceInfo>();
        public ReferenceInfo Inheritance { get; set; } = new ReferenceInfo();
        public List<ReferenceInfo> Interfaces { get; set; } = new List<ReferenceInfo>();
        public List<DependencyReference> LocalDependencies { get; set; } = new List<DependencyReference>();
        public List<DependencyReference> ForeignDependencies { get; set; } = new List<DependencyReference>();
        public Dictionary<string, DependencySummary> ForeignDependenciesByDeclaringType { get; set; } =
            new Dictionary<string, DependencySummary>();
        public List<EnviedTypeCandidate> LikelyEnviedTypeCandidates { get; set; } =
            new List<EnviedTypeCandidate>();
        public List<DependencyReference> DependencyNeighborhood { get; set; } =
            new List<DependencyReference>();
        public List<UnresolvedDependency> UnresolvedDependencies { get; set; } =
            new List<UnresolvedDependency>();
        public List<string> WorkspaceDiagnostics { get; set; } = new List<string>();
    }

    internal sealed class RepositoryInfo
    {
        public string RepositoryIdentifier { get; set; } = string.Empty;
        public string Commit { get; set; } = string.Empty;
        public string ProjectIdentifier { get; set; } = string.Empty;
        public string ProjectPath { get; set; } = string.Empty;
        public string SourceFile { get; set; } = string.Empty;
    }

    internal sealed class TargetInfo
    {
        public string ContainingType { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public string DatasetAnnotation { get; set; } = string.Empty;
    }

    internal sealed class DependencyReference
    {
        public string Relationship { get; set; } = string.Empty;
        public string Member { get; set; } = string.Empty;
        public string DeclaringType { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Scope { get; set; } = string.Empty;
        public LocationInfo? Location { get; set; }
    }

    internal sealed class ReferenceInfo
    {
        public string Relationship { get; set; } = string.Empty;
        public string ResolutionStatus { get; set; } = string.Empty;
        public string? DeclaringType { get; set; }
        public string? Member { get; set; }
        public LocationInfo? Location { get; set; }
    }

    internal sealed class LocationInfo
    {
        public bool IsSource { get; set; }
        public string? File { get; set; }
        public int StartLine { get; set; }
        public int StartColumn { get; set; }
        public int EndLine { get; set; }
        public int EndColumn { get; set; }
        public int Start { get; set; }
        public int Length { get; set; }

        internal static LocationInfo? Create(ISymbol symbol)
        {
            var location = symbol.Locations.FirstOrDefault(candidate => candidate.IsInSource);
            if (location == null)
            {
                return new LocationInfo { IsSource = false };
            }

            var span = location.GetLineSpan();
            return new LocationInfo
            {
                IsSource = true,
                File = span.Path,
                StartLine = span.StartLinePosition.Line + 1,
                StartColumn = span.StartLinePosition.Character + 1,
                EndLine = span.EndLinePosition.Line + 1,
                EndColumn = span.EndLinePosition.Character + 1,
                Start = location.SourceSpan.Start,
                Length = location.SourceSpan.Length
            };
        }
    }

    internal sealed class DependencySummary
    {
        public string DeclaringType { get; set; } = string.Empty;
        public bool IsSourceType { get; set; }
        public int FieldAccesses { get; set; }
        public int PropertyAccesses { get; set; }
        public int MethodCalls { get; set; }
        public int TotalReferences { get; set; }
    }

    internal sealed class EnviedTypeCandidate
    {
        public string DeclaringType { get; set; } = string.Empty;
        public int Score { get; set; }
        public int FieldAccesses { get; set; }
        public int PropertyAccesses { get; set; }
        public int MethodCalls { get; set; }
    }

    internal sealed class UnresolvedDependency
    {
        public string Relationship { get; set; } = string.Empty;
        public string Expression { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
    }
}
