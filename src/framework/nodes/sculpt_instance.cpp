#include "sculpt_instance.h"

SculptInstance::SculptInstance()
{
}

SculptInstance::~SculptInstance()
{
}

std::vector<Stroke>& SculptInstance::get_stroke_history()
{
    return stroke_history;
}
