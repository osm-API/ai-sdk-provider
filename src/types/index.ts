import type { LanguageModelV3, LanguageModelV3Prompt } from '@ai-sdk/provider';

export type { LanguageModelV3, LanguageModelV3Prompt };

export * from './osm-chat-settings';
export * from './osm-completion-settings';
export * from './osm-embedding-settings';
export * from './osm-image-settings';

export type OsmProviderOptions = {
  models?: string[];

  /**
   * Optional reasoning settings
   */
  reasoning?: {
    enabled?: boolean;
    exclude?: boolean;
  } & (
    | {
        max_tokens: number;
      }
    | {
        effort: 'xhigh' | 'high' | 'medium' | 'low' | 'minimal' | 'none';
      }
  );

  /**
   * A unique identifier representing your end-user, which can
   * help OSM to monitor and detect abuse.
   */
  user?: string;
};

export type OsmSharedSettings = OsmProviderOptions & {
  /**
   * @deprecated use `reasoning` instead
   */
  includeReasoning?: boolean;

  extraBody?: Record<string, unknown>;

  /**
   * Enable usage accounting to get detailed token usage information.
   */
  usage?: {
    /**
     * When true, includes token usage information in the response.
     */
    include: boolean;
  };
};

/**
 * Usage accounting response
 */
export type OsmUsageAccounting = {
  promptTokens: number;
  promptTokensDetails?: {
    cachedTokens: number;
  };
  completionTokens: number;
  completionTokensDetails?: {
    reasoningTokens: number;
  };
  totalTokens: number;
  cost?: number;
  costDetails?: {
    upstreamInferenceCost: number;
  };
};
