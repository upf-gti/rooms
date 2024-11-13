#include "player_stage.h"

#include "engine/rooms_engine.h"

#include "framework/input.h"
#include "framework/resources/room.h"

void PlayerStage::initialize()
{

}

void PlayerStage::clean()
{

}

void PlayerStage::on_enter(void* data)
{
    current_room = reinterpret_cast<Room*>(data);
}

void PlayerStage::on_exit()
{
    if (current_room) {
        current_room->stop();
    }
}

void PlayerStage::update(float delta_time)
{
    // debug, exit player..
    if (Input::was_key_pressed(GLFW_KEY_ESCAPE) || Input::was_button_pressed(XR_BUTTON_Y)) {
        RoomsEngine::switch_stage(SCENE_EDITOR);
    }

    if (!current_room) {
        return;
    }

    current_room->update(delta_time);
}

void PlayerStage::render()
{
    if (!current_room) {
        return;
    }

    current_room->render();
}
