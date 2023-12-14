#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "graphics/renderers/rooms_renderer.h"

#include <fstream>

#include "spdlog/spdlog.h"

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    sculpt_editor.initialize();

    std::string environment = "data/textures/environments/sky.hdre";

    skybox = parse_mesh("data/meshes/cube.obj");
    skybox->set_material_shader(RendererStorage::get_shader("data/shaders/mesh_texture_cube.wgsl"));
    skybox->set_material_diffuse(RendererStorage::get_texture(environment));
    skybox->scale(glm::vec3(100.f));
    skybox->set_material_priority(2);

    //parse_scene("data/gltf_tests/DamagedHelmetGLB/DamagedHelmet.glb", entities);

    // import_scene();

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    sculpt_editor.clean();
}

void RoomsEngine::update(float delta_time)
{
    Engine::update(delta_time);

    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    skybox->set_translation(renderer->get_camera()->get_eye());

    sculpt_editor.update(delta_time);

    if (Input::was_key_pressed(GLFW_KEY_E))
    {
        export_scene();
    }
}

void RoomsEngine::render()
{
    skybox->render();

	for (auto entity : entities) {
		entity->render();
	}

    sculpt_editor.render();

	Engine::render();
}

bool RoomsEngine::export_scene()
{
    //std::ofstream file("data/exports/myscene.txt");

    //if (!file.is_open())
    //    return false;

    //// Write scene info
    //RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    //RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    //auto edits = rmr->get_scene_edits();

    //file << "@" << edits.size() << "\n";

    //glm::vec3 position = rmr->get_sculpt_start_position();
    //file << "@" << std::to_string(position.x) << " " + std::to_string(position.y) + " " + std::to_string(position.z) << "\n";

    //for (const Edit& edit : edits)
    //    file << edit.to_string() << "\n";

    //file.close();

    //spdlog::info("Scene exported! ({} edits)", edits.size());

    return true;
}

bool RoomsEngine::import_scene()
{
    std::ifstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    std::string line = "";

    // Write scene info
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    struct {
        int num_edits = 0;
        // ...
    } scene_header;

    // Num edits
    std::getline(file, line);
    scene_header.num_edits = std::stoi(line.substr(1));

    // Starting sculpt position
    std::getline(file, line);
    glm::vec3 position = load_vec3(line.substr(1));
    rmr->set_sculpt_start_position(position);
    sculpt_editor.set_sculpt_started(true);

    std::vector<Edit> edits;
    edits.resize(scene_header.num_edits);
    int edit_count = 0;

    // Parse edits
    while (std::getline(file, line))
    {
        Edit edit;
        edit.parse_string(line);
        edits[edit_count++] = edit;
    }

    file.close();

    if (edit_count != scene_header.num_edits)
    {
        spdlog::error("[import_scene] Some edits couldn't be imported!");
        return false;
    }

    // Merge them into the scene in chunks of 64

    int chunk_size = 64;
    int chunks = ceil((float)edit_count / chunk_size);

    for (int i = 0; i < chunks; ++i)
    {
        int start_index = i * chunk_size;
        int end_index = std::min(start_index + chunk_size, scene_header.num_edits);

        for (int j = start_index; j < end_index; ++j)
        {
            rmr->push_edit( edits[j] );
            edit_count--;
        }

        rmr->compute_octree();
    }

    spdlog::info("Scene imported! ({} edits, {} left)", scene_header.num_edits, edit_count);

    return true;
}
