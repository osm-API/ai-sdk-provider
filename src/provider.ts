import type { ProviderV3 } from '@ai-sdk/provider';
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
import type {
  OsmImageModelId,
  OsmImageSettings,
} from './types/osm-image-settings';

import { loadApiKey, withoutTrailingSlash } from '@ai-sdk/provider-utils';
import { OsmChatLanguageModel } from './chat';
import { OsmCompletionLanguageModel } from './completion';
import { OsmEmbeddingModel } from './embedding';
import { OsmImageModel } from './image';
import { withUserAgentSuffix } from './utils/with-user-agent-suffix';
import { VERSION } from './version';

export type { OsmChatSettings, OsmCompletionSettings };

export interface OsmProvider extends ProviderV3 {
  (
    modelId: OsmChatModelId,
    settings?: OsmCompletionSettings,
  ): OsmCompletionLanguageModel;
  (modelId: OsmChatModelId, settings?: OsmChatSettings): OsmChatLanguageModel;

  languageModel(
    modelId: OsmChatModelId,
    settings?: OsmCompletionSettings,
  ): OsmCompletionLanguageModel;
  languageModel(
    modelId: OsmChatModelId,
    settings?: OsmChatSettings,
  ): OsmChatLanguageModel;

  /**
Creates an OSM chat model for text generation.
   */
  chat(
    modelId: OsmChatModelId,
    settings?: OsmChatSettings,
  ): OsmChatLanguageModel;

  /**
Creates an OSM completion model for text generation.
   */
  completion(
    modelId: OsmCompletionModelId,
    settings?: OsmCompletionSettings,
  ): OsmCompletionLanguageModel;

  /**
Creates an OSM text embedding model. (AI SDK v5)
   */
  textEmbeddingModel(
    modelId: OsmEmbeddingModelId,
    settings?: OsmEmbeddingSettings,
  ): OsmEmbeddingModel;

  /**
Creates an OSM text embedding model. (AI SDK v4 - deprecated, use textEmbeddingModel instead)
@deprecated Use textEmbeddingModel instead
   */
  embedding(
    modelId: OsmEmbeddingModelId,
    settings?: OsmEmbeddingSettings,
  ): OsmEmbeddingModel;

  /**
Creates an OSM image model for image generation.
   */
  imageModel(
    modelId: OsmImageModelId,
    settings?: OsmImageSettings,
  ): OsmImageModel;
}

export interface OsmProviderSettings {
  /**
Base URL for the OSM API calls.
     */
  baseURL?: string;

  /**
@deprecated Use `baseURL` instead.
     */
  baseUrl?: string;

  /**
API key for authenticating requests.
     */
  apiKey?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string>;

  /**
OSM compatibility mode. Defaults to 'compatible'.
   */
  compatibility?: 'strict' | 'compatible';

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: typeof fetch;

  /**
A JSON object to send as the request body to access OSM features.
  */
  extraBody?: Record<string, unknown>;
}

/**
Create an OSM provider instance.
 */
export function createOsm(options: OsmProviderSettings = {}): OsmProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL ?? options.baseUrl) ??
    'https://api.osmapi.com/v1';

  const compatibility = options.compatibility ?? 'compatible';

  const getHeaders = () =>
    withUserAgentSuffix(
      {
        Authorization: `Bearer ${loadApiKey({
          apiKey: options.apiKey,
          environmentVariableName: 'OSM_API_KEY',
          description: 'OSM',
        })}`,
        ...options.headers,
      },
      `ai-sdk/osm/${VERSION}`,
    );

  const createChatModel = (
    modelId: OsmChatModelId,
    settings: OsmChatSettings = {},
  ) =>
    new OsmChatLanguageModel(modelId, settings, {
      provider: 'osm.chat',
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      compatibility,
      fetch: options.fetch,
      extraBody: options.extraBody,
    });

  const createCompletionModel = (
    modelId: OsmCompletionModelId,
    settings: OsmCompletionSettings = {},
  ) =>
    new OsmCompletionLanguageModel(modelId, settings, {
      provider: 'osm.completion',
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      compatibility,
      fetch: options.fetch,
      extraBody: options.extraBody,
    });

  const createEmbeddingModel = (
    modelId: OsmEmbeddingModelId,
    settings: OsmEmbeddingSettings = {},
  ) =>
    new OsmEmbeddingModel(modelId, settings, {
      provider: 'osm.embedding',
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
      extraBody: options.extraBody,
    });

  const createImageModel = (
    modelId: OsmImageModelId,
    settings: OsmImageSettings = {},
  ) =>
    new OsmImageModel(modelId, settings, {
      provider: 'osm.image',
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
      extraBody: options.extraBody,
    });

  const createLanguageModel = (
    modelId: OsmChatModelId | OsmCompletionModelId,
    settings?: OsmChatSettings | OsmCompletionSettings,
  ) => {
    if (new.target) {
      throw new Error(
        'The OSM model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId, settings as OsmChatSettings);
  };

  const provider = (
    modelId: OsmChatModelId | OsmCompletionModelId,
    settings?: OsmChatSettings | OsmCompletionSettings,
  ) => createLanguageModel(modelId, settings);

  provider.languageModel = createLanguageModel;
  provider.chat = createChatModel;
  provider.completion = createCompletionModel;
  provider.textEmbeddingModel = createEmbeddingModel;
  provider.embedding = createEmbeddingModel; // deprecated alias for v4 compatibility
  provider.imageModel = createImageModel;

  return provider as OsmProvider;
}

/**
Default OSM provider instance.
 */
export const osm = createOsm({
  compatibility: 'compatible',
});
