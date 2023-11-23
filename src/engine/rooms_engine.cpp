#include "rooms_engine.h"
#include "framework/entities/entity_mesh.h"
#include "framework/entities/entity_text.h"
#include "framework/input.h"
#include "framework/scene/parse_scene.h"
#include "graphics/renderers/rooms_renderer.h"

#include <iostream>
#include <fstream>

int RoomsEngine::initialize(Renderer* renderer, GLFWwindow* window, bool use_glfw, bool use_mirror_screen)
{
	int error = Engine::initialize(renderer, window, use_glfw, use_mirror_screen);

    sculpt_editor.initialize();

    //EntityMesh* torus = parse_mesh("data/meshes/torus/torus.obj");
    //torus->scale(glm::vec3(0.25));
    //torus->translate(glm::vec3(-1.0f, 0.0, 0.0));
    //entities.push_back(torus);

    //EntityMesh* cube = parse_mesh("data/meshes/cube/cube.obj");
    //cube->scale(glm::vec3(0.25));
    //cube->translate(glm::vec3(1.0f, 0.0, 0.0));
    //entities.push_back(cube);

    //EntityMesh* cube2 = parse_mesh("data/meshes/cube/cube.obj");
    //cube2->scale(glm::vec3(0.25));
    //cube2->translate(glm::vec3(4.0f, 0.0, 0.0));
    //entities.push_back(cube2);

    //TextEntity* text = new TextEntity("oppenheimer vs barbie");
    //text->set_material_color(colors::GREEN);
    //text->set_scale(0.25f)->generate_mesh();
    //text->translate(glm::vec3(0.0f, 0.0, -5.0));
    //entities.push_back(text);

    //parse_scene("data/gltf_tests/Sponza/Sponza.gltf", entities);

    // import_scene();

    // Create instance of HDRE
    HDRE* hdre = HDRE::Get("data/textures/environments/grass.hdre");

    if (!hdre) {
        spdlog::error("Can't load HDRE!");
        error = 1;
    }
    else
    {
        Texture* cube_texture = new Texture();
        cube_texture->load_from_hdre(hdre);

        EntityMesh* cube = parse_mesh("data/meshes/cube.obj");
        cube->set_material_shader(RendererStorage::get_shader("data/shaders/mesh_texture_cube.wgsl"));
        cube->set_material_diffuse(cube_texture);
        cube->set_material_priority(2);
        entities.push_back(cube);
    }

	return error;
}

void RoomsEngine::clean()
{
    Engine::clean();

    sculpt_editor.clean();
}

void RoomsEngine::update(float delta_time)
{
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    entities[0]->set_translation(renderer->get_camera()->get_eye());

	Engine::update(delta_time);

    sculpt_editor.update(delta_time);

    if (Input::was_key_pressed(GLFW_KEY_E))
    {
        export_scene();
    }
}

void RoomsEngine::render()
{
	for (auto entity : entities) {
		entity->render();
	}

    sculpt_editor.render();

	Engine::render();
}

bool RoomsEngine::export_scene()
{
    std::ofstream file("data/exports/myscene.txt");

    if (!file.is_open())
        return false;

    // Write scene info
    RoomsRenderer* renderer = static_cast<RoomsRenderer*>(RoomsRenderer::instance);
    RaymarchingRenderer* rmr = renderer->get_raymarching_renderer();

    auto edits = rmr->get_scene_edits();

    file << "@" << edits.size() << "\n";

    glm::vec3 position = rmr->get_sculpt_start_position();
    file << "@" << std::to_string(position.x) << " " + std::to_string(position.y) + " " + std::to_string(position.z) << "\n";

    for (const Edit& edit : edits)
        file << edit.to_string() << "\n";

    file.close();

    spdlog::info("Scene exported! ({} edits)", edits.size());

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
