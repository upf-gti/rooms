#pragma once
#include "ui/ui_widgets.h"
#include "framework/colors.h"
#include <variant>
#include <functional>
#include <map>
#include <string>
#include <vector>

class RaymarchingRenderer;
class EntityMesh;

using FuncFloat = std::function<void(const std::string&, float)>;
using FuncString = std::function<void(const std::string&, std::string)>;
using FuncVec2 = std::function<void(const std::string&, glm::vec2)>;
using FuncVec3 = std::function<void(const std::string&, glm::vec3)>;
using FuncVec4 = std::function<void(const std::string&, glm::vec4)>;

using SignalType = std::variant < FuncFloat, FuncString, FuncVec2, FuncVec3, FuncVec4>;

namespace ui {

    const float BUTTON_SIZE         = 32.f;
    const float X_MARGIN            = 8.f;
    const float X_GROUP_MARGIN      = X_MARGIN * 0.5f;
    const float Y_MARGIN            = 12.f;

	struct WorkSpaceData {
		glm::vec2 size;
		uint8_t select_button;
		uint8_t root_pose;
		uint8_t hand;
		uint8_t select_hand;
	};

	class Controller {

        RaymarchingRenderer* renderer = nullptr;
		WorkSpaceData workspace;
		glm::mat4x4 global_transform;

		EntityMesh* raycast_pointer = nullptr;

		ui::Widget* root = nullptr;
		std::map<std::string, std::vector<SignalType>> signals;

        static std::map<std::string, Widget*> groups;

		/*
		*	Widget Helpers
		*/

        bool group_opened = false;
        float g_iterator = 0.f;

        glm::vec2 layout_iterator = { 0.f, 0.f };
        glm::vec2 last_layout_pos;
		std::vector<Widget*> parent_queue;

		void append_widget( Widget* widget );
		void process_params(glm::vec2& position, glm::vec2& size, bool skip_to_local = false);
        glm::vec2 compute_position();

	public:

		float global_scale = 1.f;
		bool enabled = true;

		/*
		*	Select button: XR Buttons
		*	Root pose: AIM, GRIP
		*	Hand: To set UI panel
		*	Select hand: Raycast hand
		*/

		const WorkSpaceData& get_workspace() { return workspace; };
		void set_workspace(glm::vec2 _workspace_size, uint8_t _select_button = 0, uint8_t _root_pose = 1, uint8_t _hand = 0, uint8_t _select_hand = 1);
		const glm::mat4x4& get_matrix() { return global_transform; };
		bool is_active();

		void render();
		void update(float delta_time);

		/*
		*	Widgets
		*/

        Widget* make_rect(glm::vec2 pos, glm::vec2 size, const Color& color);
		Widget* make_text(const std::string& text, glm::vec2 pos, const Color& color, float scale = 1.f, glm::vec2 size = {1, 1});
		Widget* make_button(const std::string& signal, const char* texture = nullptr, const char* shader = "data/shaders/mesh_texture_ui.wgsl", const Color& color = colors::WHITE);
		Widget* make_slider(const std::string& signal, float default_value, glm::vec2 pos, glm::vec2 size, const Color& color, const char* texture = nullptr);
		Widget* make_color_picker(const std::string& signal, const Color& default_color, glm::vec2 pos, glm::vec2 size);
        void make_submenu(Widget* parent, const std::string& name);
        void close_submenu();

        Widget* make_group(const std::string& group_name, float number_of_widgets, const Color& color = colors::WHITE);
        void close_group();

        static Widget* get_group_from_alias(const std::string& alias);

		/*
		*	Callbacks
		*/

		void bind(const std::string& name, SignalType callback);

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
