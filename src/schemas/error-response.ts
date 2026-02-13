import type { ChatErrorError } from '../types/osm-api-types';

import { createJsonErrorResponseHandler } from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

export const osmErrorResponseSchema = z
  .object({
    error: z
      .object({
        code: z
          .union([z.string(), z.number()])
          .nullable()
          .optional()
          .default(null),
        message: z.string(),
        type: z.string().nullable().optional().default(null),
        param: z.any().nullable().optional().default(null),
      })
      .passthrough() satisfies z.ZodType<
      Omit<ChatErrorError, 'code'> & { code: string | number | null }
    >,
  })
  .passthrough();

export type osmErrorData = z.infer<typeof osmErrorResponseSchema>;

export const osmFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: osmErrorResponseSchema,
  errorToMessage: (data: osmErrorData) => data.error.message,
});
