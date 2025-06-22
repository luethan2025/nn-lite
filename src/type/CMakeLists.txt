add_library(
        focus_type
        OBJECT
        float_tensor.cpp)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:focus_type>
        PARENT_SCOPE)
