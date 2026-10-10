// Compositor layer effects — GLSL translations of the Swift MSL shaders.
//
// These are reference implementations of what the GPU path will look like
// once we wire Qt RHI / QQuickItem + QSGShaderEffect. The CPU path in
// ``layer_effects.py`` already produces equivalent output; this file
// exists so the API contract and naming matches.
//
// Conventions:
//   * Coordinates are in canvas pixels.
//   * sampler2D srcTex is the layer's RGBA buffer.
//   * sampler2D maskTex is the layer mask (1 channel, white = visible).
//
// Each effect has a ``uniforms`` block listing its Photoshop-equivalent
// parameters. Driver defaults reproduce the no-op case.

uniform float u_dpi;            // for unit conversion (1 px = 1/dp at 1.0)

// ============================================================================
// Drop Shadow
// ============================================================================
#ifdef EFFECT_DROP_SHADOW
uniform vec3  u_color;          // shadow tint
uniform float u_opacity;        // 0..1
uniform float u_angle;          // degrees, clockwise from +x
uniform float u_distance;       // pixels
uniform float u_blur;           // std-dev in pixels
uniform float u_spread;         // 0..1 — grow/shrink the alpha before blur
uniform sampler2D maskTex;      // optional mask

vec4 dropShadow(vec2 uv, vec4 src) {
    float a = radians(u_angle);
    vec2 offset = vec2(cos(a), sin(a)) * u_distance / vec2(textureSize(srcTex, 0));
    float dist = length((uv - offset - vec2(0.5)) * vec2(textureSize(srcTex, 0)));
    float shadow = src.a * smoothstep(u_blur, 0.0, dist) * u_opacity;
    // Spread biases the threshold.
    shadow = clamp(shadow + u_spread - 0.5, 0.0, 1.0);
    return vec4(u_color * shadow, shadow);
}
#endif

// ============================================================================
// Inner Shadow
// ============================================================================
#ifdef EFFECT_INNER_SHADOW
uniform vec3  u_color;
uniform float u_opacity;
uniform float u_angle;
uniform float u_distance;
uniform float u_blur;
uniform float u_choke;          // 0..1 — choke = spread inverted

vec4 innerShadow(vec2 uv, vec4 src) {
    vec4 neighbor = texture(srcTex, uv);
    // Walk inward from the alpha boundary by u_distance pixels.
    float a = src.a;
    float inside = smoothstep(0.0, u_blur, u_distance / 100.0);
    float shadow = (1.0 - a) * inside * u_opacity;
    return vec4(u_color * shadow, shadow);
}
#endif

// ============================================================================
// Outer Glow
// ============================================================================
#ifdef EFFECT_OUTER_GLOW
uniform vec3  u_color;
uniform float u_opacity;
uniform float u_blur;
uniform float u_spread;

vec4 outerGlow(vec2 uv, vec4 src) {
    vec4 n = texture(srcTex, uv + vec2(u_blur) / vec2(textureSize(srcTex, 0)));
    float glow = (1.0 - n.a) * u_opacity * smoothstep(0.0, u_blur, 0.5);
    return vec4(u_color * glow, glow);
}
#endif

// ============================================================================
// Inner Glow
// ============================================================================
#ifdef EFFECT_INNER_GLOW
uniform vec3  u_color;
uniform float u_opacity;
uniform float u_blur;

vec4 innerGlow(vec2 uv, vec4 src) {
    float inside = (1.0 - src.a) * u_opacity;
    inside *= smoothstep(0.0, u_blur, 0.5);
    return vec4(u_color * inside, inside);
}
#endif

// ============================================================================
// Stroke
// ============================================================================
#ifdef EFFECT_STROKE
uniform vec3  u_color;
uniform float u_opacity;
uniform float u_size;           // pixels
uniform int   u_position;       // 0 inside, 1 outside, 2 centre

vec4 stroke(vec2 uv, vec4 src) {
    vec2 texel = 1.0 / vec2(textureSize(srcTex, 0));
    float max_a = src.a;
    // Sample a 3×3 ring around the current pixel to find the edge.
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec4 n = texture(srcTex, uv + vec2(dx, dy) * texel * u_size);
            max_a = max(max_a, n.a);
        }
    }
    float edge = max_a - src.a;
    return vec4(u_color * edge * u_opacity, edge * u_opacity);
}
#endif

// ============================================================================
// Color Overlay
// ============================================================================
#ifdef EFFECT_COLOR_OVERLAY
uniform vec3  u_color;
uniform float u_opacity;

vec4 colorOverlay(vec2 uv, vec4 src) {
    return vec4(mix(src.rgb, u_color, u_opacity), src.a);
}
#endif
