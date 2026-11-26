#include "haptics_manager.h"

#include "spdlog/spdlog.h"
#include <fstream>
#include <rapidjson/document.h>
#include "framework/utils/json_utils.h"

// Networking sockets
#ifdef _WIN32
bool Socket::socket_dll_loaded = false;
#endif

#define HAPTICS_LOAD_CONFIG_FROM_FILE 1

HapticsManager::HapticsManager() {
#ifdef HAPTICS_LOAD_CONFIG_FROM_FILE
    rapidjson::Document config = load_json("haptics_config.json");

    if (config.IsObject()) {
        max_temperature = config["max_temperature"].GetFloat();
        min_temperature = config["min_temperature"].GetFloat();
        max_rumble = config["max_rumble"].GetFloat();
        min_rumble = config["min_rumble"].GetFloat();

        rumble_strength = config["rumble_strength"].GetFloat();

        max_haptic_speed = config["haptics_speed_cap"].GetFloat();
        haptic_speed_min_multiplayer = config["haptics_speed_min_multipliyer"].GetFloat();
        haptic_speed_max_multiplayer = config["haptics_speed_max_multipliyer"].GetFloat();
        rumble_difference = config["rumble_difference"].GetInt();

        haptics_url = config["haptic_device_IP"].GetString();
        listening_port = config["haptic_device_port"].GetInt();

        spdlog::info("HAPTICS: Loaded config haptic file.");
    } else {
        spdlog::info("HAPTICS: Could not find the haptics config file.");
    }
#endif
    spdlog::info("HAPTICS: temperature range: {} - {} C", min_temperature, max_temperature);
    spdlog::info("HAPTICS: rumble range: {} - {} Hz", min_rumble, max_rumble);
    spdlog::info("HAPTCIS: rumble strength {}", rumble_strength);
    spdlog::info("HAPTICS: Connection dir: {} Port: {}", haptics_url, listening_port);

    set_connection(haptics_url, listening_port);

    std::string message_to_send = message_generator(0, 0, 0, 0);
    last_sent_message = 0u;
    if (!haptic_socket.send_to((const char*)message_to_send.c_str(), sizeof(char) * message_to_send.length())) {
        spdlog::error("HAPTICS: Error sending first message");
    }
}

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
    return haptic_socket.CreateClientSocket(hapt_url, std::to_string(listening_port), &error_code, false);
}

void HapticsManager::update_contact_info(const bool contact, const float mat_roughness, const float mat_metalness, const glm::vec3 hand_speed)
{
    current_status.contact = contact;

    float capped_speed = glm::min(glm::length(hand_speed), max_haptic_speed);
    // If the speed is 0, then the frequency should be 0
    float speed_mult_frequency = std::lerp(haptic_speed_min_multiplayer, haptic_speed_max_multiplayer, capped_speed / max_haptic_speed) * glm::sign(capped_speed);
    // Map of roughnes to rumble Hzs
    current_status.rumble = (uint8_t) glm::min(std::lerp(min_rumble, max_rumble, mat_roughness) * speed_mult_frequency, 255.0f);
    // A full metalic surface should give cold feedback, so close, to 0
    current_status.temperature = (uint8_t) std::lerp(max_temperature, min_temperature, mat_metalness);
}


bool HapticsManager::send_update() {
    // Not send message if there is no change to the haptics
    if (old_status.contact == current_status.contact &&
        old_status.rumble == current_status.rumble &&
        old_status.temperature == current_status.temperature &&
        (last_sent_message == 2u || last_sent_message == 0u)) {
        // NOTE: UN-COMENT THIS IF YOU WANT TO AVOID SENDING A LOT OF MESSAGES:
        // Since it is UDP and the message can get lost, maybe this is better?
        // If not, we can send each message a certain set of times
        //return true;
    }

    uint8_t HAPTICS_TEMP_DELTA = 4u; // Boundary to signal when a temperature change is significative enought
    uint8_t HAPTICS_RUMBLE_DELTA = 1u; // Boundary to signal when a rumble change is significative enought
    std::string message_to_send = message_generator(0, 0, 0, 0);
    last_sent_message = 0u;

    if (current_status.contact) {
        if (!old_status.contact || abs(current_status.temperature - old_status.temperature) >  HAPTICS_TEMP_DELTA) {
            // Config message. Since PBR paramter changes are very subtle, we set the power to a third of 255 (the max)
            message_to_send = message_generator(1, current_status.temperature, rumble_strength, rumble_strength);
            last_sent_message = 1u;
        } else {
            // Update message
            // 90 degrees to the motor that pulls the finger
            // Set slightly different rumbles to each motor
            message_to_send = message_generator(2, glm::max(0u, current_status.rumble - rumble_difference), current_status.rumble, 90);
            last_sent_message = 2u;
        }
    }

    spdlog::info("HAPTICS: Send message: {}", message_to_send);

    old_status = current_status;
    return haptic_socket.send_to((const char*) message_to_send.c_str(), sizeof(char) * message_to_send.length());
}
