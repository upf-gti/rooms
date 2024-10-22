#include "player_editor.h"

#include "framework/resources/room.h"

void PlayerEditor::initialize()
{
    
}

void PlayerEditor::clean()
{

}

void PlayerEditor::on_enter(void* data)
{
    
}

void PlayerEditor::on_exit()
{
    if (current_room) {
        current_room->stop();
    }
}

void PlayerEditor::update(float delta_time)
{
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
