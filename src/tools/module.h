#pragma once

class RoomsRenderer;

class Module {

protected:

    RoomsRenderer* renderer = nullptr;

public:

    Module() {};

    virtual void initialize() {};
    virtual void clean() {};

    virtual void update(float delta_time) {};
    virtual void render() {};
    virtual void render_gui() {};
};
