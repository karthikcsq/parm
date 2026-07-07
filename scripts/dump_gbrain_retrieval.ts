import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

function argument(name: string): string {
  const position = process.argv.indexOf(name);
  const value = position >= 0 ? process.argv[position + 1] : undefined;
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function vectorToArray(value: unknown): number[] {
  if (Array.isArray(value)) {
    return value.map(Number);
  }
  if (ArrayBuffer.isView(value)) {
    return Array.from(value as unknown as Iterable<number>, Number);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    const body =
      (trimmed.startsWith("[") && trimmed.endsWith("]")) ||
      (trimmed.startsWith("(") && trimmed.endsWith(")"))
        ? trimmed.slice(1, -1)
        : trimmed;
    if (!body.trim()) return [];
    const vector = body.split(",").map((item) => Number(item.trim()));
    if (vector.some((item) => !Number.isFinite(item))) {
      throw new Error("PGLite returned an invalid vector string");
    }
    return vector;
  }
  throw new Error(`Unsupported embedding value: ${typeof value}`);
}

const runtime = resolve(argument("--runtime"));
const configModule = await import(
  pathToFileURL(resolve(runtime, "src/core/config.ts")).href
);
const factoryModule = await import(
  pathToFileURL(resolve(runtime, "src/core/engine-factory.ts")).href
);
const config = configModule.loadConfig();
if (!config) throw new Error("GBrain is not configured");
const engine = await factoryModule.createEngine(
  configModule.toEngineConfig(config),
);

await engine.connect(configModule.toEngineConfig(config));
try {
  const pages = await engine.executeRaw(`
    SELECT id, source_id, slug, title, frontmatter
    FROM pages
    WHERE deleted_at IS NULL
    ORDER BY source_id, slug
  `);
  const chunks = await engine.executeRaw(`
    SELECT c.id, c.page_id, c.chunk_index, c.chunk_text, c.embedding, c.model,
           p.embedding_signature
    FROM content_chunks c
    JOIN pages p ON p.id = c.page_id
    WHERE p.deleted_at IS NULL
      AND c.modality = 'text'
      AND c.embedding IS NOT NULL
    ORDER BY p.source_id, p.slug, c.chunk_index, c.id
  `);
  const links = await engine.executeRaw(`
    SELECT l.from_page_id, l.to_page_id, l.link_type, l.link_source
    FROM links l
    JOIN pages source ON source.id = l.from_page_id
    JOIN pages target ON target.id = l.to_page_id
    WHERE source.deleted_at IS NULL AND target.deleted_at IS NULL
    ORDER BY l.from_page_id, l.to_page_id, l.id
  `);
  const normalizedChunks = chunks.map((chunk: Record<string, unknown>) => ({
    ...chunk,
    embedding: vectorToArray(chunk.embedding),
  }));
  process.stdout.write(
    JSON.stringify({
      config: {
        embedding_model: config.embedding_model,
        embedding_dimensions: config.embedding_dimensions,
      },
      pages,
      chunks: normalizedChunks,
      links,
    }),
  );
} finally {
  await engine.disconnect();
}
