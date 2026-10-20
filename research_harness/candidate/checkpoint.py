"""Reference checkpoint API. Единственная разрешённая поверхность solver patch."""


def save(memory):
    return memory.export_state()


def restore(memory_class, serialized):
    return memory_class.from_state(serialized)
