import type { FileAnnotation } from '@/src/schemas/provider-metadata';
import type { ReasoningDetailUnion } from '@/src/schemas/reasoning-details';

// Type for Osm Cache Control following Anthropic's pattern
export type OsmCacheControl = { type: 'ephemeral' };

export type OsmChatCompletionsInput = Array<ChatCompletionMessageParam>;

export type ChatCompletionMessageParam =
  | ChatCompletionSystemMessageParam
  | ChatCompletionUserMessageParam
  | ChatCompletionAssistantMessageParam
  | ChatCompletionToolMessageParam;

export interface ChatCompletionSystemMessageParam {
  role: 'system';
  content: Array<ChatCompletionContentPartText>;
}

export interface ChatCompletionUserMessageParam {
  role: 'user';
  content: string | Array<ChatCompletionContentPart>;
  cache_control?: OsmCacheControl;
}

export type ChatCompletionContentPart =
  | ChatCompletionContentPartText
  | ChatCompletionContentPartImage
  | ChatCompletionContentPartFile
  | ChatCompletionContentPartInputAudio;

export interface ChatCompletionContentPartFile {
  type: 'file';
  file: {
    filename?: string;
    file_data?: string;
    file_id?: string;
  };
  cache_control?: OsmCacheControl;
}

export interface ChatCompletionContentPartImage {
  type: 'image_url';
  image_url: {
    url: string;
  };
  cache_control?: OsmCacheControl;
}

export interface ChatCompletionContentPartText {
  type: 'text';
  text: string;
  reasoning?: string | null;
  cache_control?: OsmCacheControl;
}

/** https://osm.ai/docs/guides/overview/multimodal/audio */
export const OSM_AUDIO_FORMATS = [
  'wav',
  'mp3',
  'aiff',
  'aac',
  'ogg',
  'flac',
  'm4a',
  'pcm16',
  'pcm24',
] as const;

export type OsmAudioFormat = (typeof OSM_AUDIO_FORMATS)[number];

export interface ChatCompletionContentPartInputAudio {
  type: 'input_audio';
  input_audio: {
    data: string;
    format: OsmAudioFormat;
  };
  cache_control?: OsmCacheControl;
}

export interface ChatCompletionAssistantMessageParam {
  role: 'assistant';
  content?: string | null;
  reasoning?: string | null;
  reasoning_details?: ReasoningDetailUnion[];
  annotations?: FileAnnotation[];
  tool_calls?: Array<ChatCompletionMessageToolCall>;
  cache_control?: OsmCacheControl;
}

export interface ChatCompletionMessageToolCall {
  type: 'function';
  id: string;
  function: {
    arguments: string;
    name: string;
  };
}

export interface ChatCompletionToolMessageParam {
  role: 'tool';
  content: string;
  tool_call_id: string;
  cache_control?: OsmCacheControl;
}
