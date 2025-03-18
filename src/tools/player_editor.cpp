#include "player_editor.h"

#include "engine/rooms_engine.h"

#include "framework/input.h"
#include "framework/resources/room.h"

void PlayerEditor::initialize()
{
    
}

void PlayerEditor::clean()
{

}

void PlayerEditor::on_enter(void* data)
{
    RoomsEngine* engine = static_cast<RoomsEngine*>(RoomsEngine::instance);
    engine->hide_controllers();

    current_room = reinterpret_cast<Room*>(data);
}

void PlayerEditor::on_exit()
{
    if (current_room) {
        current_room->stop();
    }
}

void PlayerEditor::update(float delta_time)
{
    // debug, exit player..
    if (Input::was_key_pressed(GLFW_KEY_ESCAPE) || Input::was_button_pressed(XR_BUTTON_Y)) {
        RoomsEngine::switch_editor(SCENE_EDITOR);
    }

    if (!current_room) {
        return;
    }

    current_room->update(delta_time);
}

void PlayerEditor::render()
{
    if (!current_room) {
        return;
    }

    current_room->render();
}
