#include "haptics_manager.h"

#include "spdlog/spdlog.h"

// Networking sockets
#ifdef _WIN32
bool Socket::socket_dll_loaded = false;
#endif

HapticsManager::~HapticsManager()
{
    haptic_socket.close();
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
    current_status.roughness = mat_roughness;
    current_status.metalness = mat_metalness;
}


bool HapticsManager::send_update() {
    std::string message_to_send = message_generator(0, 0, 0, 0);

    if (current_status.contact) {
        if (old_status.contact) {
            // Config message
            message_to_send = message_generator(1, 0, 0, 0);
        } else {
            // Update message
            // TODO: do this, but good 
            message_to_send = message_generator(2, (uint8_t) current_status.roughness * 255.0, (uint8_t)(1.0f - current_status.metalness) * 255.0, 0);
        }
        
        spdlog::trace("Send Haptic respose: {}", message_to_send);
    }

    old_status = current_status;
    return haptic_socket.send_to((const char*) message_to_send.c_str(), sizeof(char) * message_to_send.length());
}
