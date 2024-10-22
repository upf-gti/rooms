#include "room.h"

#include "engine/scene.h"

void Room::start()
{
    running = true;
}

void Room::stop()
{
    running = false;
    reset();
}

void Room::reset()
{
    // ...
}

void Room::update(float delta_time)
{
    if (!running) {
        return;
    }

    if (scene) {
        scene->update(delta_time);
    }
}

void Room::render()
{
    if (scene) {
        scene->render();
    }
}

bool Room::load(const std::string& load_path)
{
    if (scene) {
        scene->parse(load_path);
        return true;
    }

    return false;
}

void Room::save(const std::string& save_path)
{
    if (scene) {
        scene->serialize(save_path);
    }
}
