#include "scene_editor.h"

#include "includes.h"

#include "framework/nodes/viewport_3d.h"
#include "framework/input.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "engine/rooms_engine.h"
#include "engine/scene.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

Gizmo3D SceneEditor::gizmo = {};

void SceneEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    main_scene = Engine::instance->get_main_scene();

    gizmo.initialize(POSITION_GIZMO, { 0.0f, 0.0f, 0.0f });
}

void SceneEditor::clean()
{
    gizmo.clean();
}

void SceneEditor::update(float delta_time)
{
    Node3D* node = (Node3D*)main_scene->get_nodes().back();

    glm::vec3 right_controller_pos = Input::get_controller_position(HAND_RIGHT, POSE_AIM);
    Transform t = mat4ToTransform(node->get_model());

    if (gizmo.update(t, right_controller_pos, delta_time)) {
        node->set_transform(t);
    }
}

void SceneEditor::render()
{
    gizmo.render();

    static_cast<RoomsEngine*>(RoomsEngine::instance)->render_controllers();
}

void SceneEditor::render_gui()
{

}
