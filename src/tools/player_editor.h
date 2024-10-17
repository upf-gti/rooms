#pragma once

#include "base_editor.h"

class PlayerEditor : public BaseEditor {
public:

    PlayerEditor() {};
    PlayerEditor(const std::string& name) : BaseEditor(name) {};

    void initialize() override;
    void clean() override;

    void update(float delta_time) override;
    void render() override;

    void on_enter(void* data);
    void on_exit();
};
