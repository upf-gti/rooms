#pragma once

#define OCTREE_DEPTH 6
#define ATLAS_BRICK_SIZE 8u
#define ATLAS_BRICK_NO_BORDER_SIZE (ATLAS_BRICK_SIZE - 2u)
#define SSAA_SDF_WRITE_TO_TEXTURE false
#define PREVIEW_EDITS_MAX 128
#define SDF_RESOLUTION 400
#define SCULPT_MAX_SIZE 1 // meters
#define PREVIEW_PROXY_BRICKS_COUNT 10000u

#define MIN_SMOOTH_FACTOR 0.0001f
#define MAX_SMOOTH_FACTOR 0.02f

#define MIN_PRIMITIVE_SIZE 0.002f
#define MAX_PRIMITIVE_SIZE 0.25f

#define PREVIEW_BASE_EDIT_LIST 200u
#define PREVIEW_EDIT_LIST_INCREMENT 200u

#define EDIT_BUFFER_INITIAL_SIZE 256u
#define EDIT_BUFFER_INCREASE 256u
#define STROKE_CONTEXT_INITIAL_SIZE 100u
#define STROKE_CONTEXT_INCREASE 100u
#define AREA_MAX_EVALUATION_SIZE  (1.0f / 4.0f)

// TODO: use 16 bit stroke indices for culling -> double the culling capacity
#define MAX_STROKE_INFLUENCE_COUNT 240u
