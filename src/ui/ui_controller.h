#pragma once
#include "ui/ui_widgets.h"
#include "framework/colors.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

class EntityMesh;

namespace ui {

	struct WorkSpaceData {
		glm::vec2 size;
		uint8_t select_button;
		uint8_t root_pose;
		uint8_t hand;
		uint8_t select_hand;
	};

	class Controller {

		WorkSpaceData workspace;
		glm::mat4x4 global_transform;

		EntityMesh* raycast_pointer = nullptr;
		EntityMesh* workspace_element = nullptr;

		std::vector <ui::Widget*> root;
		std::map <std::string, std::function<void(const std::string&, float)>> signals;

	public:

		float global_scale = 1.f;

		/*
		*	Select button: XR Buttons
		*	Root pose: AIM, GRIP
		*	Hand: To set UI panel
		*	Select hand: Raycast hand
		*/

		const WorkSpaceData& get_workspace() { return workspace; };
		void set_workspace(glm::vec2 _workspace_size, uint8_t _select_button = 0, uint8_t _root_pose = 0, uint8_t _hand = 0, uint8_t _select_hand = 1);
		const glm::mat4x4& get_matrix() { return global_transform; };
		bool is_active();

		void render();
		void update(float delta_time);

		/*
		*	Widgets
		*/

		void make_text(const std::string& text, glm::vec2 pos, const glm::vec3& color, float scale = 1.f, glm::vec2 size = {1, 1});
		void make_button(const std::string& signal, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture = nullptr);
		void make_slider(const std::string& signal, glm::vec2 pos, glm::vec2 size, const glm::vec3& color, const char* texture = nullptr);

		/*
		*	Callbacks
		*/

		void connect(const std::string& name, std::function<void(const std::string&, float)> callback);
		bool emit_signal(const std::string& name, float value = 0.f);
	};
}