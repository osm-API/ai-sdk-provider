import { z } from 'zod/v4';

const osmEmbeddingUsageSchema = z.object({
  prompt_tokens: z.number(),
  total_tokens: z.number(),
  cost: z.number().optional(),
});

const osmEmbeddingDataSchema = z.object({
  object: z.literal('embedding'),
  embedding: z.array(z.number()),
  index: z.number().optional(),
});

export const OsmEmbeddingResponseSchema = z.object({
  id: z.string().optional(),
  object: z.literal('list'),
  data: z.array(osmEmbeddingDataSchema),
  model: z.string(),
  provider: z.string().optional(),
  usage: osmEmbeddingUsageSchema.optional(),
});

export type OsmEmbeddingResponse = z.infer<typeof OsmEmbeddingResponseSchema>;
