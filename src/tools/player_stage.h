#pragma once

#include "stage.h"

class Room;

class PlayerStage : public Stage {

    Room* current_room = nullptr;

public:

    PlayerStage() {};
    PlayerStage(const std::string& name) : Stage(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    void on_enter(void* data);
    void on_exit();
};
