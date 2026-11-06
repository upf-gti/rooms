#pragma once

#include <string>
#include <stdint.h>
#include <glm/glm.hpp>

// Networking sockets
#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

#else // __linux__ and MACOS
#include <sys/socket.h>

#endif

struct Socket {
    bool initialized = false;

#ifdef _WIN32
    static bool socket_dll_loaded;
    SOCKET socket_connection = INVALID_SOCKET; // = int32*
#else // __linux__ and MACOS

#endif

    bool CreateClientTCPSocket(const std::string address, const std::string listening_port, int* error_code) {
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
            .ai_protocol = IPPROTO_TCP
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
        
        return true;
#else // __linux__ and MACOS
        assert(false);
        // TODO
#endif
    }

    bool listen_to(char* msg_buffer, const uint32_t max_msg_buffer_size, uint32_t* msg_length) {
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

#endif
    }
    bool close() {
#ifdef _WIN32
        int32_t result_code = shutdown(socket_connection, SD_SEND);

        closesocket(socket_connection);
        socket_connection = INVALID_SOCKET;

        return result_code != SOCKET_ERROR;
#else // __linux__ and MACOS
        TODO
#endif
    }

    bool clean_sockets() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

// Networking sockets
#ifdef _WIN32

bool Socket::socket_dll_loaded = false;
#endif

class HapticsManager {
    uint32_t listening_port;
    std::string haptics_url;

    struct sHapticStatus {
        bool contact = false;
        float roughness = 0.0f; // Up to 1.0f
        float metalness = 0.0f; // Up to 1.0f
    } current_status;


    HapticsManager();

    // NOTE: the roughnes is only percived if the hand is moving. The rumble depends on hand speed * mat_roughness
    //       Metalness does not, only on contact.
    void update_contact_info(const bool contact, const float mat_roughness, const float mat_metalness, const glm::vec3 hand_speed);
};
