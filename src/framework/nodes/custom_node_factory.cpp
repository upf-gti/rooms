#include "custom_node_factory.h"

#include "framework/nodes/default_node_factory.h"

#include "sculpt_instance.h"

Node* custom_node_factory(const std::string& node_type)
{
    Node* node = nullptr;

    if (node_type == "SculptInstance") {
        node = new SculptInstance();
    }

    if (!node) {
        node = default_node_factory(node_type);
    }

    return node;
}
