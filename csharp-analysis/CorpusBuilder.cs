using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.MSBuild;

namespace CSharpDependencyAnalyzer
{
    internal static class CorpusBuilder
    {
        internal static async Task<int> RunAsync(string[] args)
        {
            try
            {
                var options = CorpusOptions.Parse(args);
                var documents = await BuildAsync(options);
                WriteJsonLines(options.OutputPath, documents);
                Console.Error.WriteLine("METHOD_DOCUMENTS: " + documents.Count);
                Console.Error.WriteLine("UNIQUE_DOCUMENT_IDS: " +
                    documents.Select(document => document.DocumentId).Distinct(StringComparer.Ordinal).Count());
                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception.Message);
                return 1;
            }
        }

        internal static async Task<int> RunDependenciesAsync(string[] args)
        {
            try
            {
                var options = CorpusOptions.Parse(args);
                var documents = await BuildDependenciesAsync(options);
                WriteJsonLines(options.OutputPath, documents);
                Console.Error.WriteLine("METHOD_DEPENDENCY_DOCUMENTS: " + documents.Count);
                Console.Error.WriteLine("UNIQUE_DOCUMENT_IDS: " +
                    documents.Select(document => document.DocumentId).Distinct(StringComparer.Ordinal).Count());
                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception.Message);
                return 1;
            }
        }

        private static void WriteJsonLines<T>(string outputPath, List<T> documents)
        {
            var outputDirectory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
            if (!string.IsNullOrEmpty(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            using (var writer = new StreamWriter(outputPath, false, new UTF8Encoding(false)))
            {
                foreach (var document in documents)
                {
                    writer.WriteLine(JsonSerializer.Serialize(document));
                }
            }
        }

        private static async Task<List<CorpusDocument>> BuildAsync(CorpusOptions options)
        {
            RegisterMsBuild();

            using (var workspace = MSBuildWorkspace.Create())
            {
                var project = await workspace.OpenProjectAsync(
                    Path.GetFullPath(options.ProjectPath),
                    cancellationToken: CancellationToken.None);
                var documents = new List<CorpusDocument>();

                foreach (var document in project.Documents
                    .Where(document => IsEligibleSource(document.FilePath))
                    .OrderBy(document => document.FilePath, StringComparer.OrdinalIgnoreCase))
                {
                    var root = await document.GetSyntaxRootAsync();
                    var semanticModel = await document.GetSemanticModelAsync();
                    if (root == null || semanticModel == null || document.FilePath == null)
                    {
                        continue;
                    }

                    foreach (var method in root.DescendantNodes().OfType<MethodDeclarationSyntax>())
                    {
                        var symbol = semanticModel.GetDeclaredSymbol(method) as IMethodSymbol;
                        if (symbol == null)
                        {
                            continue;
                        }

                        var relativePath = Path.GetRelativePath(
                            Path.GetFullPath(options.RepositoryRoot),
                            Path.GetFullPath(document.FilePath));
                        var methodName = GetMethodName(method, symbol, semanticModel);
                        var signature = GetSignature(method, symbol, methodName);
                        var documentId = CreateDocumentId(
                            options, relativePath, symbol, signature);
                        var corpusDocument = new CorpusDocument
                        {
                            DocumentId = documentId,
                            Repository = options.RepositoryIdentifier,
                            Commit = options.Commit,
                            Project = project.Name,
                            File = NormalizePath(relativePath),
                            Namespace = symbol.ContainingNamespace?.ToDisplayString() ?? string.Empty,
                            ContainingType = symbol.ContainingType.ToDisplayString(),
                            MethodName = methodName,
                            Signature = signature,
                            ParameterTypes = symbol.Parameters
                                .Select(parameter => parameter.Type.ToDisplayString())
                                .ToList(),
                            ReturnType = symbol.ReturnType.ToDisplayString(),
                            StartLine = method.GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                            EndLine = method.GetLocation().GetLineSpan().EndLinePosition.Line + 1,
                            SourceCode = method.ToFullString()
                        };
                        corpusDocument.RetrievalText = string.Join(
                            "\n",
                            corpusDocument.Namespace,
                            corpusDocument.ContainingType,
                            corpusDocument.Signature,
                            corpusDocument.SourceCode);
                        documents.Add(corpusDocument);
                    }
                }

                return documents
                    .OrderBy(document => document.File, StringComparer.Ordinal)
                    .ThenBy(document => document.StartLine)
                    .ThenBy(document => document.Signature, StringComparer.Ordinal)
                    .ToList();
            }
        }

        private static async Task<List<MethodDependencyDocument>> BuildDependenciesAsync(
            CorpusOptions options)
        {
            RegisterMsBuild();

            using (var workspace = MSBuildWorkspace.Create())
            {
                var project = await workspace.OpenProjectAsync(
                    Path.GetFullPath(options.ProjectPath),
                    cancellationToken: CancellationToken.None);
                var documents = new List<MethodDependencyDocument>();

                foreach (var document in project.Documents
                    .Where(document => IsEligibleSource(document.FilePath))
                    .OrderBy(document => document.FilePath, StringComparer.OrdinalIgnoreCase))
                {
                    var root = await document.GetSyntaxRootAsync();
                    var semanticModel = await document.GetSemanticModelAsync();
                    if (root == null || semanticModel == null || document.FilePath == null)
                    {
                        continue;
                    }

                    var relativePath = Path.GetRelativePath(
                        Path.GetFullPath(options.RepositoryRoot),
                        Path.GetFullPath(document.FilePath));
                    foreach (var method in root.DescendantNodes().OfType<MethodDeclarationSyntax>())
                    {
                        var symbol = semanticModel.GetDeclaredSymbol(method) as IMethodSymbol;
                        if (symbol == null)
                        {
                            continue;
                        }

                        var methodName = GetMethodName(method, symbol, semanticModel);
                        var signature = GetSignature(method, symbol, methodName);
                        var documentId = CreateDocumentId(
                            options, relativePath, symbol, signature);
                        var collector = new DependencyCollector(
                            semanticModel,
                            method,
                            symbol.ContainingType);
                        collector.Collect();
                        documents.Add(new MethodDependencyDocument
                        {
                            SchemaVersion = "1.0-semantic-method-dependencies",
                            DocumentId = documentId,
                            Repository = options.RepositoryIdentifier,
                            Commit = options.Commit,
                            File = NormalizePath(relativePath),
                            ContainingType = symbol.ContainingType.ToDisplayString(),
                            MethodName = methodName,
                            Signature = signature,
                            StartLine = method.GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                            FieldAccesses = collector.Fields,
                            PropertyAccesses = collector.Properties,
                            UnresolvedReferences = collector.Unresolved
                        });
                    }
                }

                return documents
                    .OrderBy(document => document.File, StringComparer.Ordinal)
                    .ThenBy(document => document.Signature, StringComparer.Ordinal)
                    .ToList();
            }
        }

        private static string CreateDocumentId(
            CorpusOptions options,
            string relativePath,
            IMethodSymbol symbol,
            string signature)
        {
            var canonicalIdInput = string.Join(
                "\n",
                options.RepositoryIdentifier,
                options.Commit,
                NormalizePath(relativePath),
                symbol.ContainingType.ToDisplayString(),
                signature);
            return StableId(canonicalIdInput);
        }

        private static string GetMethodName(
            MethodDeclarationSyntax method,
            IMethodSymbol symbol,
            SemanticModel semanticModel)
        {
            if (method.ExplicitInterfaceSpecifier == null)
            {
                return symbol.Name;
            }

            var interfaceName = semanticModel.GetTypeInfo(
                method.ExplicitInterfaceSpecifier.Name).Type?.ToDisplayString() ??
                method.ExplicitInterfaceSpecifier.Name.ToString();
            return interfaceName + "." + method.Identifier.ValueText;
        }

        private static string GetSignature(
            MethodDeclarationSyntax method,
            IMethodSymbol symbol,
            string methodName)
        {
            if (method.ExplicitInterfaceSpecifier == null)
            {
                return symbol.ToDisplayString();
            }

            var normalSignature = symbol.ToDisplayString();
            var normalMethodSuffix = "." + method.Identifier.ValueText + "(";
            var suffixIndex = normalSignature.LastIndexOf(
                normalMethodSuffix,
                StringComparison.Ordinal);
            if (suffixIndex < 0)
            {
                return normalSignature;
            }

            return normalSignature.Substring(0, suffixIndex) +
                "." + methodName +
                normalSignature.Substring(suffixIndex + normalMethodSuffix.Length - 1);
        }

        private static void RegisterMsBuild()
        {
            if (MSBuildLocator.IsRegistered)
            {
                return;
            }

            var msbuildPath = Environment.GetEnvironmentVariable("ROSLYN_MSBUILD_PATH");
            if (string.IsNullOrWhiteSpace(msbuildPath))
            {
                throw new InvalidOperationException(
                    "Set ROSLYN_MSBUILD_PATH to the local MSBuild SDK directory.");
            }

            MSBuildLocator.RegisterMSBuildPath(msbuildPath);
        }

        private static bool IsEligibleSource(string? path)
        {
            if (string.IsNullOrWhiteSpace(path) ||
                !path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            var segments = path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (segments.Any(segment =>
                string.Equals(segment, "bin", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(segment, "obj", StringComparison.OrdinalIgnoreCase)))
            {
                return false;
            }

            var fileName = Path.GetFileName(path);
            return !fileName.EndsWith(".g.cs", StringComparison.OrdinalIgnoreCase) &&
                !fileName.EndsWith(".g.i.cs", StringComparison.OrdinalIgnoreCase) &&
                !fileName.EndsWith(".designer.cs", StringComparison.OrdinalIgnoreCase);
        }

        private static string StableId(string value)
        {
            using (var sha256 = SHA256.Create())
            {
                var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(value));
                return "cs-method-" + Convert.ToHexString(bytes).ToLowerInvariant();
            }
        }

        private static string NormalizePath(string path)
        {
            return path.Replace(Path.DirectorySeparatorChar, '/').Replace('\\', '/');
        }
    }

    internal sealed class CorpusOptions
    {
        internal string ProjectPath { get; private set; } = string.Empty;
        internal string RepositoryRoot { get; private set; } = string.Empty;
        internal string RepositoryIdentifier { get; private set; } = string.Empty;
        internal string Commit { get; private set; } = string.Empty;
        internal string OutputPath { get; private set; } = string.Empty;

        internal static CorpusOptions Parse(string[] args)
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

            return new CorpusOptions
            {
                ProjectPath = Required(values, "project"),
                RepositoryRoot = Required(values, "repository-root"),
                RepositoryIdentifier = Required(values, "repository"),
                Commit = Required(values, "commit"),
                OutputPath = Required(values, "output")
            };
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

    internal sealed class CorpusDocument
    {
        public string DocumentId { get; set; } = string.Empty;
        public string Repository { get; set; } = string.Empty;
        public string Commit { get; set; } = string.Empty;
        public string Project { get; set; } = string.Empty;
        public string File { get; set; } = string.Empty;
        public string Namespace { get; set; } = string.Empty;
        public string ContainingType { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public List<string> ParameterTypes { get; set; } = new List<string>();
        public string ReturnType { get; set; } = string.Empty;
        public int StartLine { get; set; }
        public int EndLine { get; set; }
        public string SourceCode { get; set; } = string.Empty;
        public string RetrievalText { get; set; } = string.Empty;
    }

    internal sealed class MethodDependencyDocument
    {
        public string SchemaVersion { get; set; } = string.Empty;
        public string DocumentId { get; set; } = string.Empty;
        public string Repository { get; set; } = string.Empty;
        public string Commit { get; set; } = string.Empty;
        public string File { get; set; } = string.Empty;
        public string ContainingType { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public int StartLine { get; set; }
        public List<DependencyReference> FieldAccesses { get; set; } =
            new List<DependencyReference>();
        public List<DependencyReference> PropertyAccesses { get; set; } =
            new List<DependencyReference>();
        public List<UnresolvedDependency> UnresolvedReferences { get; set; } =
            new List<UnresolvedDependency>();
    }
}
