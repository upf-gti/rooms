#pragma once

#include "base_editor.h"

class Room;

class PlayerEditor : public BaseEditor {

    Room* current_room = nullptr;

public:

    PlayerEditor() {};
    PlayerEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    void on_enter(void* data) override;
    void on_exit() override;
};
