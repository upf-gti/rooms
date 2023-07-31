#pragma once
#include "ui/ui_widgets.h"
#include "framework/colors.h"
#include <variant>
#include <functional>
#include <map>
#include <string>
#include <vector>

class EntityMesh;

using FuncFloat = std::function<void(const std::string&, float)>;
using FuncString = std::function<void(const std::string&, std::string)>;
using FuncVec2 = std::function<void(const std::string&, glm::vec2)>;
using FuncVec3 = std::function<void(const std::string&, glm::vec3)>;
using FuncVec4 = std::function<void(const std::string&, glm::vec4)>;

using SignalType = std::variant < FuncFloat, FuncString, FuncVec2, FuncVec3, FuncVec4>;

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
		std::map <std::string, std::vector<SignalType>> signals;

		/*
		*	Widget Helpers
		*/

		void process_params(glm::vec2& position, glm::vec2& size, bool skip_to_local = false);
		Widget* make_rect(glm::vec2 pos, glm::vec2 size, const Color& color);

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

		Widget* make_text(const std::string& text, glm::vec2 pos, const Color& color, float scale = 1.f, glm::vec2 size = {1, 1});
		Widget* make_button(const std::string& signal, glm::vec2 pos, glm::vec2 size, const Color& color, const char* texture = nullptr);
		Widget* make_slider(const std::string& signal, float default_value, glm::vec2 pos, glm::vec2 size, const Color& color, const char* texture = nullptr);
		Widget* make_color_picker(const std::string& signal, const Color& default_color, glm::vec2 pos, glm::vec2 size);

		/*
		*	Callbacks
		*/

		void connect(const std::string& name, SignalType callback);

		template<typename T>
		bool emit_signal(const std::string& name, T value) {

			auto it = signals.find(name);
			if (it == signals.end())
				return false;

			using FuncT = std::function<void(const std::string&, T)>;

			for (auto& f : signals[name])
			{
				if (std::holds_alternative<FuncT>(f))
					std::get<FuncT>(f)(name, value);
			}

			return true;
		}
	};
}