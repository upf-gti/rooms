#include "haptics_manager.h"

#include "spdlog/spdlog.h"

// Networking sockets
#ifdef _WIN32
bool Socket::socket_dll_loaded = false;
#endif

HapticsManager::~HapticsManager()
{
    haptic_socket.close_socket();
}

bool HapticsManager::set_connection(const std::string hapt_url,
                                    const uint32_t list_ports) {
    haptics_url = hapt_url;
    listening_port = list_ports;

    // TODO: Managet this gracely
    int32_t error_code;
    return haptic_socket.CreateClientSocket(hapt_url, std::to_string(listening_port), &error_code, true);
}

void HapticsManager::update_contact_info(const bool contact, const float mat_roughness, const float mat_metalness, const glm::vec3 hand_speed)
{
    current_status.contact = contact;
    // TODO: this magic constants should be loaded from a text file
    // Map of roughnes to rumble Hzs
    current_status.rumble = (uint8_t) std::lerp(0.0f, 150.0f, mat_roughness);
    // A full metalic surface should give cold feedback, so close, to 0
    current_status.temperature = (uint8_t) std::lerp(25.0f, 5.0f, mat_metalness);
}


bool HapticsManager::send_update() {
    // Not send message if there is no change to the haptics
    if (old_status.contact == current_status.contact &&
        old_status.rumble == current_status.rumble &&
        old_status.temperature == current_status.temperature &&
        (last_sent_message == 2u || last_sent_message == 0u)) {
        return true;
    }

    uint8_t HAPTICS_TEMP_DELTA = 4u; // Boundary to signal when a temperature change is significative enought
    uint8_t HAPTICS_RUMBLE_DELTA = 1u; // Boundary to signal when a rumble change is significative enought
    std::string message_to_send = message_generator(0, 0, 0, 0);
    last_sent_message = 0u;

    if (current_status.contact) {
        if (!old_status.contact || abs(current_status.temperature - old_status.temperature) >  HAPTICS_TEMP_DELTA) {
            // Config message. Since PBR paramter changes are very subtle, we set the power to a third of 255 (the max)
            message_to_send = message_generator(1, current_status.temperature, 85, 85);
            last_sent_message = 1u;
        } else {
            // Update message
            // 90 degrees to the motor that pulls the finger
            message_to_send = message_generator(2, current_status.rumble, current_status.rumble, 90);
            last_sent_message = 2u;
        }
    }

    spdlog::info("Send Haptic respose: {}", message_to_send);

    old_status = current_status;
    return haptic_socket.send_to((const char*) message_to_send.c_str(), sizeof(char) * message_to_send.length());
}
