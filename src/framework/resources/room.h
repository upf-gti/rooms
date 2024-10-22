#pragma once

#include "framework/resources/resource.h"

#include <vector>

class Scene;

class Room : public Resource {

    uint32_t uid = 0u;

    Scene* scene = nullptr;

    bool running = false;

    void reset();

public:

    Room() {};

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
