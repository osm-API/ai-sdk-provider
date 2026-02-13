import type { OsmProviderSettings } from './provider';
import type {
  OsmChatModelId,
  OsmChatSettings,
} from './types/osm-chat-settings';
import type {
  OsmCompletionModelId,
  OsmCompletionSettings,
} from './types/osm-completion-settings';
import type {
  OsmEmbeddingModelId,
  OsmEmbeddingSettings,
} from './types/osm-embedding-settings';

import { loadApiKey, withoutTrailingSlash } from '@ai-sdk/provider-utils';
import { OsmChatLanguageModel } from './chat';
import { OsmCompletionLanguageModel } from './completion';
import { OsmEmbeddingModel } from './embedding';

/**
@deprecated Use `createOsm` instead.
 */
export class Osm {
  /**
Use a different URL prefix for API calls, e.g. to use proxy servers.
The default prefix is `https://osm.ai/api/v1`.
   */
  readonly baseURL: string;

  /**
API key that is being sent using the `Authorization` header.
It defaults to the `OPENROUTER_API_KEY` environment variable.
 */
  readonly apiKey?: string;

  /**
Custom headers to include in the requests.
   */
  readonly headers?: Record<string, string>;

  /**
   * Creates a new Osm provider instance.
   */
  constructor(options: OsmProviderSettings = {}) {
    this.baseURL =
      withoutTrailingSlash(options.baseURL ?? options.baseUrl) ??
      'https://api.osmapi.com/v1';
    this.apiKey = options.apiKey;
    this.headers = options.headers;
  }

  private get baseConfig() {
    return {
      baseURL: this.baseURL,
      headers: () => ({
        Authorization: `Bearer ${loadApiKey({
          apiKey: this.apiKey,
          environmentVariableName: 'OSM_API_KEY',
          description: 'Osm',
        })}`,
        ...this.headers,
      }),
    };
  }

  chat(modelId: OsmChatModelId, settings: OsmChatSettings = {}) {
    return new OsmChatLanguageModel(modelId, settings, {
      provider: 'osm.chat',
      ...this.baseConfig,
      compatibility: 'strict',
      url: ({ path }) => `${this.baseURL}${path}`,
    });
  }

  completion(
    modelId: OsmCompletionModelId,
    settings: OsmCompletionSettings = {},
  ) {
    return new OsmCompletionLanguageModel(modelId, settings, {
      provider: 'osm.completion',
      ...this.baseConfig,
      compatibility: 'strict',
      url: ({ path }) => `${this.baseURL}${path}`,
    });
  }

  textEmbeddingModel(
    modelId: OsmEmbeddingModelId,
    settings: OsmEmbeddingSettings = {},
  ) {
    return new OsmEmbeddingModel(modelId, settings, {
      provider: 'osm.embedding',
      ...this.baseConfig,
      url: ({ path }) => `${this.baseURL}${path}`,
    });
  }

  /**
   * @deprecated Use textEmbeddingModel instead
   */
  embedding(modelId: OsmEmbeddingModelId, settings: OsmEmbeddingSettings = {}) {
    return this.textEmbeddingModel(modelId, settings);
  }
}
