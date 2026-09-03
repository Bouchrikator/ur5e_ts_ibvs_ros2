cmake_minimum_required(VERSION 3.22)

# Optimus 1.1 (upstream sofa-framework/Optimus @ cd41bf17) ported to SOFA v25.12.
# Only the stochastic-filtering core used by the cable identification pipeline is
# built: FilteringAnimationLoop, ROUKFilter, StochasticStateWrapper, OptimParams,
# MappedStateObservationManager, SimulatedStateObservationSource,
# PreStochasticWrapper (+ FilterEvents, TimeProfiling). See UPSTREAM.md.
project(Optimus VERSION 1.1 LANGUAGES CXX)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.DefaultType REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.LinearAlgebra REQUIRED)
sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Simulation.Common REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(STOCHASTIC_FILTERING "Build Optimus stochastic filtering" ON)
option(USE_BLAS_FOR_OPTIMUS "Use BLAS (dgemm) instead of Eigen in ROUKFilter" OFF)

set(OPTIMUS_HEADERS
    src/optimusConfig.h
    src/initOptimusPlugin.h
    src/genericComponents/TimeProfiling.h
    src/genericComponents/FilterEvents.h
    src/genericComponents/ObservationSource.h
    src/genericComponents/OptimParams.h
    src/genericComponents/OptimParams.inl
    src/genericComponents/SimulatedStateObservationSource.h
    src/genericComponents/SimulatedStateObservationSource.inl
)
set(OPTIMUS_SOURCES
    src/initOptimusPlugin.cpp
    src/genericComponents/FilterEvents.cpp
    src/genericComponents/OptimParams.cpp
    src/genericComponents/SimulatedStateObservationSource.cpp
)
if(STOCHASTIC_FILTERING)
    list(APPEND OPTIMUS_HEADERS
        src/stochasticFiltering/StochasticFilterBase.h
        src/stochasticFiltering/StochasticStateWrapperBase.h
        src/stochasticFiltering/ObservationManagerBase.h
        src/stochasticFiltering/FilteringAnimationLoop.h
        src/stochasticFiltering/PreStochasticWrapper.h
        src/stochasticFiltering/StochasticStateWrapper.h
        src/stochasticFiltering/StochasticStateWrapper.inl
        src/stochasticFiltering/ROUKFilter.h
        src/stochasticFiltering/ROUKFilter.inl
        src/stochasticFiltering/MappedStateObservationManager.h
        src/stochasticFiltering/MappedStateObservationManager.inl
    )
    list(APPEND OPTIMUS_SOURCES
        src/stochasticFiltering/FilteringAnimationLoop.cpp
        src/stochasticFiltering/PreStochasticWrapper.cpp
        src/stochasticFiltering/StochasticStateWrapper.cpp
        src/stochasticFiltering/ROUKFilter.cpp
        src/stochasticFiltering/MappedStateObservationManager.cpp
    )
endif()

add_library(${PROJECT_NAME} SHARED ${OPTIMUS_HEADERS} ${OPTIMUS_SOURCES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_OPTIMUSPLUGIN)
# Installed at <SOFA_ROOT>/plugins/Optimus/lib: resolve the SOFA component libraries
# that are not yet loaded when SofaPython3/runSofa dlopen() the plugin.
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../../lib")
if(STOCHASTIC_FILTERING)
    target_compile_definitions(${PROJECT_NAME} PUBLIC OPTIMUS_STOCHASTIC_FILTERING)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/Optimus>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Sofa.Core
        Sofa.DefaultType
        Sofa.Helper
        Sofa.LinearAlgebra
        Sofa.Simulation.Core
        Sofa.Simulation.Common
        Sofa.Component.StateContainer
        Sofa.Component.Constraint.Projective
        Sofa.Component.Constraint.Lagrangian.Solver
        Sofa.Component.Topology.Container.Dynamic
        Eigen3::Eigen
)

if(USE_BLAS_FOR_OPTIMUS)
    find_package(BLAS REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE BLAS::BLAS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE BLAS_LINKED)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
)
