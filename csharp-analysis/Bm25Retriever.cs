using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace CSharpDependencyAnalyzer
{
    internal static class Bm25Retriever
    {
        private const double K1 = 1.5;
        private const double B = 0.75;
        private static readonly Regex TokenPattern =
            new Regex("[A-Za-z0-9]+", RegexOptions.Compiled);

        internal static int Run(string[] args)
        {
            try
            {
                var options = Bm25Options.Parse(args);
                var documents = File.ReadLines(options.CorpusPath)
                    .Where(line => !string.IsNullOrWhiteSpace(line))
                    .Select(line => JsonSerializer.Deserialize<CorpusDocument>(line)
                        ?? throw new InvalidDataException("Corpus line deserialized to null."))
                    .ToList();

                var target = documents.Single(document =>
                    string.Equals(document.DocumentId, options.TargetDocumentId, StringComparison.Ordinal));
                var query = BuildQuery(target);
                var index = new Bm25Index(documents);
                var results = index.Search(query, target.DocumentId, 10);
                var output = new Bm25Result
                {
                    CorpusPath = Path.GetFullPath(options.CorpusPath),
                    CorpusDocumentCount = documents.Count,
                    QueryTargetDocumentId = target.DocumentId,
                    QueryTargetSignature = target.Signature,
                    QueryText = query,
                    K1 = K1,
                    B = B,
                    TopK = 10,
                    Results = results
                };

                File.WriteAllText(
                    options.OutputPath,
                    JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
                Console.Error.WriteLine("CORPUS_DOCUMENTS: " + documents.Count);
                Console.Error.WriteLine("RESULTS: " + results.Count);
                Console.Error.WriteLine("OUTPUT: " + Path.GetFullPath(options.OutputPath));
                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception.Message);
                return 1;
            }
        }

        internal static string BuildQuery(CorpusDocument target)
        {
            return string.Join(
                "\n",
                target.Namespace,
                target.ContainingType,
                target.Signature,
                target.SourceCode);
        }

        private sealed class Bm25Index
        {
            private readonly List<IndexedDocument> _documents;
            private readonly Dictionary<string, int> _documentFrequency;
            private readonly double _averageDocumentLength;

            internal Bm25Index(IReadOnlyCollection<CorpusDocument> documents)
            {
                _documents = documents.Select(document => new IndexedDocument(document)).ToList();
                _documentFrequency = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var document in _documents)
                {
                    foreach (var term in document.TermFrequency.Keys)
                    {
                        _documentFrequency[term] = _documentFrequency.TryGetValue(term, out var count)
                            ? count + 1
                            : 1;
                    }
                }

                _averageDocumentLength = _documents.Count == 0
                    ? 0
                    : _documents.Average(document => document.Length);
            }

            internal List<Bm25Hit> Search(string query, string excludedDocumentId, int topK)
            {
                var queryTerms = Tokenize(query).Distinct(StringComparer.Ordinal).ToList();
                var scored = new List<Bm25Hit>();
                foreach (var document in _documents)
                {
                    if (string.Equals(document.Source.DocumentId, excludedDocumentId, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    var score = 0d;
                    foreach (var term in queryTerms)
                    {
                        if (!document.TermFrequency.TryGetValue(term, out var termFrequency) ||
                            !_documentFrequency.TryGetValue(term, out var documentFrequency))
                        {
                            continue;
                        }

                        var inverseDocumentFrequency = Math.Log(
                            1d + (_documents.Count - documentFrequency + 0.5d) /
                            (documentFrequency + 0.5d));
                        var denominator = termFrequency +
                            K1 * (1d - B + B * document.Length / _averageDocumentLength);
                        score += inverseDocumentFrequency *
                            (termFrequency * (K1 + 1d) / denominator);
                    }

                    scored.Add(new Bm25Hit
                    {
                        Rank = 0,
                        Score = score,
                        DocumentId = document.Source.DocumentId,
                        Repository = document.Source.Repository,
                        File = document.Source.File,
                        ContainingType = document.Source.ContainingType,
                        MethodName = document.Source.MethodName,
                        Signature = document.Source.Signature,
                        StartLine = document.Source.StartLine,
                        EndLine = document.Source.EndLine
                    });
                }

                return scored
                    .OrderByDescending(hit => hit.Score)
                    .ThenBy(hit => hit.DocumentId, StringComparer.Ordinal)
                    .Take(topK)
                    .Select((hit, index) =>
                    {
                        hit.Rank = index + 1;
                        return hit;
                    })
                    .ToList();
            }
        }

        private sealed class IndexedDocument
        {
            internal IndexedDocument(CorpusDocument source)
            {
                Source = source;
                var tokens = Tokenize(source.RetrievalText).ToList();
                Length = tokens.Count;
                TermFrequency = tokens
                    .GroupBy(token => token, StringComparer.Ordinal)
                    .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
            }

            internal CorpusDocument Source { get; }
            internal int Length { get; }
            internal Dictionary<string, int> TermFrequency { get; }
        }

        private static IEnumerable<string> Tokenize(string text)
        {
            foreach (Match match in TokenPattern.Matches(text))
            {
                var value = match.Value;
                foreach (var part in Regex.Split(value, "(?<!^)(?=[A-Z])"))
                {
                    if (!string.IsNullOrWhiteSpace(part))
                    {
                        yield return part.ToLowerInvariant();
                    }
                }
            }
        }
    }

    internal sealed class Bm25Options
    {
        internal string CorpusPath { get; private set; } = string.Empty;
        internal string TargetDocumentId { get; private set; } = string.Empty;
        internal string OutputPath { get; private set; } = string.Empty;

        internal static Bm25Options Parse(string[] args)
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

            return new Bm25Options
            {
                CorpusPath = Required(values, "corpus"),
                TargetDocumentId = Required(values, "target-id"),
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

    internal sealed class Bm25Result
    {
        public string CorpusPath { get; set; } = string.Empty;
        public int CorpusDocumentCount { get; set; }
        public string QueryTargetDocumentId { get; set; } = string.Empty;
        public string QueryTargetSignature { get; set; } = string.Empty;
        public string QueryText { get; set; } = string.Empty;
        public double K1 { get; set; }
        public double B { get; set; }
        public int TopK { get; set; }
        public List<Bm25Hit> Results { get; set; } = new List<Bm25Hit>();
    }

    internal sealed class Bm25Hit
    {
        public int Rank { get; set; }
        public double Score { get; set; }
        public string DocumentId { get; set; } = string.Empty;
        public string Repository { get; set; } = string.Empty;
        public string File { get; set; } = string.Empty;
        public string ContainingType { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string Signature { get; set; } = string.Empty;
        public int StartLine { get; set; }
        public int EndLine { get; set; }
    }
}
