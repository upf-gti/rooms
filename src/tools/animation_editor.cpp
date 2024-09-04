#include "animation_editor.h"

#include "framework/utils/utils.h"
#include "framework/input.h"
#include "framework/scene/parse_obj.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/nodes/animation_player.h"

#include "graphics/renderers/rooms_renderer.h"
#include "graphics/renderer_storage.h"

#include "engine/rooms_engine.h"

#include "spdlog/spdlog.h"
#include "imgui.h"

Node3D* root = nullptr;
MeshInstance3D* node = nullptr;
AnimationPlayer* player = nullptr;

void AnimationEditor::initialize()
{
    renderer = dynamic_cast<RoomsRenderer*>(Renderer::instance);

    root = new Node3D();

    node = new MeshInstance3D();

    parse_obj("data/meshes/sphere.obj", node);

    node->set_name("node_0");

    Material* cube_material = new Material();
    cube_material->set_color(colors::PURPLE);
    cube_material->set_type(MATERIAL_PBR);
    cube_material->set_shader(RendererStorage::get_shader("data/shaders/mesh_pbr.wgsl"));

    node->set_surface_material_override(node->get_surface(0), cube_material);

    root->add_child(node);

    player = new AnimationPlayer("Animation Player");

    root->add_child(player);
}

void AnimationEditor::clean()
{

}

void AnimationEditor::update(float delta_time)
{
    root->update(delta_time);
}

void AnimationEditor::render()
{
    root->render();

    {
        static glm::mat4x4 test_model = node->get_model();
        Camera* camera = RoomsRenderer::instance->get_camera();
        bool changed = gizmo.render(camera->get_view(), camera->get_projection(), test_model);

        if (changed)
        {
            // node->set_model(test_model);
        }
    }
}

void AnimationEditor::render_gui()
{
    if (animation)
    {
        ImGui::Text("Animation %s", animation->get_name().c_str());
        ImGui::Text("Num Tracks %d", animation->get_track_count());

        if (ImGui::Button("Create keyframe")) {

            uint32_t num_keys = current_track->size();
            current_track->resize(num_keys + 1);

            Keyframe& frame = current_track->get_keyframe(num_keys);

            frame.time = current_time;

            frame.in = 0.0f;
            frame.value = node->get_translation();
            frame.out = 0.0f;

            current_time += 0.5f;

            animation->recalculate_duration();
        }

        if (current_track) {

            ImGui::Text("Num Keyframes %d", current_track->size());
            for (uint32_t i = 0; i < current_track->size(); ++i) {
                const Keyframe& key = current_track->get_keyframe(i);
                const glm::vec3& value = std::get<glm::vec3>(key.value);
                ImGui::Text("%d (%f) [%f %f %f]", i, key.time, value.x, value.y, value.z);
            }
        }
        else {
            if (ImGui::Button("Create position track")) {

                current_track = animation->add_track();
                current_track->set_name("translation");
                current_track->set_path(node->get_name() + "/" + "translation");
            }
        }

        if (ImGui::Button("Play")) {

            player->play("anim_test");
        }
    }
    else {
        if (ImGui::Button("Create animation")) {
            animation = new Animation();
            animation->set_name("anim_test");
            RendererStorage::register_animation(animation->get_name(), animation);
        }
    }

    auto engine = dynamic_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->show_tree_recursive(root);
}
