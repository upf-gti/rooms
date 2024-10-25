#pragma once

#include "framework/resources/resource.h"

#include <vector>

class Scene;

class Room : public Resource {

    // temporal uids
    static uint32_t last_uid;

    uint32_t uid = 0u;

    Scene* scene = nullptr;

    bool running = false;

    void reset();

public:

    Room();
    Room(Scene* new_scene);

    uint32_t get_uid() const { return uid; }

    void start();
    void stop();
    bool is_running() const { return running; }

    void update(float delta_time);
    void render();

    bool load(const std::string& load_path);
    void save(const std::string& save_path);

    // void share();
};
