#pragma once

#include <cstddef>
#include <string>
#include <stdint.h>
#include <glm/glm.hpp>

// Networking sockets
#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

#else // __linux__ and MACOS
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

struct Socket {
    bool initialized = false;

#ifdef _WIN32
    static bool socket_dll_loaded;
    SOCKET socket_connection = INVALID_SOCKET; // = int32*
#else // __linux__ and MACOS
    int socket = 0;
#endif

    bool CreateClientSocket(const std::string address, const std::string listening_port, int* error_code, bool is_TCP = true) {
#ifdef _WIN32
        // Test if the DLL is loaded (Windows only)
        if (!socket_dll_loaded) {
            WSADATA wsaData;
            int32_t result = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (result != 0) { // TODO: handle this better, in a way that also shows errors for the connection
                *error_code = result;
                return false;
            }
            socket_dll_loaded = true;
        }

        // Test that the connection is sound
        addrinfo* result = nullptr;
        addrinfo connection_hints = {
            .ai_family = AF_UNSPEC, // For IPV6
            .ai_socktype = SOCK_STREAM,
            .ai_protocol = (is_TCP) ? IPPROTO_TCP : IPPROTO_UDP
        };

        int32_t result_code = getaddrinfo(address.c_str(),
                                         listening_port.c_str(),
                                         &connection_hints,
                                         &result);

        if (result_code != 0) {
            return false;
        }

        // Create a socket
        socket_connection = socket( result->ai_family,
                                    result->ai_socktype,
                                    result->ai_protocol);

        if (socket_connection == INVALID_SOCKET) {
            freeaddrinfo(result);
            return false;
        }

        // Stablish the connection
        result_code = connect(socket_connection, result->ai_addr, (int32_t) result->ai_addrlen);
        freeaddrinfo(result);

        if (result_code == SOCKET_ERROR) {
            closesocket(socket_connection);
            socket_connection = INVALID_SOCKET;
            return false;
        }

        initialized = true;
        return true;
#else // __linux__ and MACOS
        socket = socket(AF_INET, (is_TCP) ? SOCK_STREAM : SOCK_DGRAM, 0);

        if (socket < 0) {
            return false;
        }

        sockaddr_in server_address = {
            .sin_family = AF_INET6,
            .sin_port = htons(std::stoi(listening_port)),
        };

        if (inet_pton(AF_INET, "127.0.0.1", &server_address.sin_addr) <= 0) {
            // Invalid adress
            return false;
        }

        if (connect(socket, (sockaddr*) &server_address, sizeof(sockaddr_in)) < 0) {
            return false;
        }

        initialized = true;
        return true;
#endif
    }

    bool send_to(const char* msg_buffer, const uint32_t msg_length) const {
        if (!initialized) {
            return false;
        }
#ifdef _WIN32
        int32_t result = send(socket_connection, msg_buffer, msg_length, 0u);

        return result != SOCKET_ERROR;
#else // __linux__ and MACOS
        return msg_length  == send(socket, msg_buffer, msg_length, 0);
#endif
    }

    bool listen_to(char* msg_buffer, const uint32_t max_msg_buffer_size, uint32_t* msg_length) {
        if (!initialized) {
            return false;
        }
#ifdef _WIN32
        int32_t result_code = 0;
        do {
            result_code = recv(socket_connection, msg_buffer, max_msg_buffer_size, 0);
        } while (result_code != 0);

        // Reception error
        // Note: 0 means connection closed. Is it special case?
        if (result_code <= 0) {
            return false;
        }

        *msg_length = result_code;

        return false;
#else // __linux__ and MACOS
        // TODO
#endif
    }

    bool close_socket() {
        if (!initialized) {
            return false;
        }
#ifdef _WIN32
        int32_t result_code = shutdown(socket_connection, SD_SEND);

        closesocket(socket_connection);
        socket_connection = INVALID_SOCKET;

        initialized = false;

        return result_code != SOCKET_ERROR;
#else // __linux__ and MACOS
        close(socket);
        return true;
#endif
    }

    bool clean_sockets() {
        if (!initialized) {
            return false;
        }
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

class HapticsManager {
public:
    Socket haptic_socket;

    uint32_t listening_port;
    std::string haptics_url;

    struct sHapticStatus {
        bool contact = false;
        uint8_t rumble = 0;
        uint8_t temperature = 0;
    };

    sHapticStatus current_status;
    sHapticStatus old_status;

    uint8_t last_sent_message = 0u;

    // Interpolation parameter
    float max_temperature = 25.0f;
    float min_temperature = 5.0f;

    float max_rumble = 150.0f;
    float min_rumble = 0.0f;

    inline std::string message_generator(const uint8_t message_code, const uint8_t p1, const uint8_t p2, const uint8_t p3) {
        return std::to_string(message_code) + ',' + std::to_string(p1) + ',' + std::to_string(p2) + ',' + std::to_string(p3);
    }
    HapticsManager();
    ~HapticsManager();

    bool set_connection(const std::string hapt_url, const uint32_t list_ports);

    // NOTE: the roughnes is only percived if the hand is moving. The rumble depends on hand speed * mat_roughness
    //       Metalness does not, only on contact.
    void update_contact_info(const bool contact, const float mat_roughness, const float mat_metalness, const glm::vec3 hand_speed);

    bool send_update();
};
