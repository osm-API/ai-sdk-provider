/**
 * Regression test for GitHub issue #392
 * https://github.com/osm-API/ai-sdk-provider/issues/392
 *
 * Issue: "Add support for auto-router plugin"
 *
 * Issue thread timeline:
 * - Feb 4, 2026: User requests support for the `auto-router` plugin
 *   to configure allowed models when using `openrouter/auto`.
 */
import { generateText, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createOsm } from '@/src';

vi.setConfig({
  testTimeout: 60_000,
});

describe('Issue #392: auto-router plugin support', () => {
  const osm = createOsm({
    apiKey: process.env.OSM_API_KEY,
    baseUrl: `${process.env.OSM_API_BASE}/api/v1`,
  });

  it('should generate text with openrouter/auto and auto-router plugin with allowed_models', async () => {
    const model = osm('openrouter/auto', {
      plugins: [
        {
          id: 'auto-router',
          allowed_models: ['anthropic/*', 'openai/*'],
        },
      ],
      usage: { include: true },
    });

    const response = await generateText({
      model,
      prompt: 'What is 2+2? Reply with just the number.',
    });

    expect(response.text).toBeDefined();
    expect(response.text.length).toBeGreaterThan(0);
    expect(response.usage.totalTokens).toBeGreaterThan(0);
    expect(response.finishReason).toBeDefined();
  });

  it('should generate text with openrouter/auto and auto-router plugin without allowed_models', async () => {
    const model = osm('openrouter/auto', {
      plugins: [{ id: 'auto-router' }],
      usage: { include: true },
    });

    const response = await generateText({
      model,
      prompt: 'What is 2+2? Reply with just the number.',
    });

    expect(response.text).toBeDefined();
    expect(response.text.length).toBeGreaterThan(0);
    expect(response.usage.totalTokens).toBeGreaterThan(0);
    expect(response.finishReason).toBeDefined();
  });

  it('should stream text with openrouter/auto and auto-router plugin with allowed_models', async () => {
    const model = osm('openrouter/auto', {
      plugins: [
        {
          id: 'auto-router',
          allowed_models: ['anthropic/*', 'openai/*'],
        },
      ],
      usage: { include: true },
    });

    const response = streamText({
      model,
      prompt: 'What is 2+2? Reply with just the number.',
    });

    const result = await response.text;
    expect(result).toBeDefined();
    expect(result.length).toBeGreaterThan(0);
  });
});
