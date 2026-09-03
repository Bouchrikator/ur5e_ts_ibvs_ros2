/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "initOptimusPlugin.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa
{

namespace component
{

namespace container
{
extern void registerOptimParams(sofa::core::ObjectFactory* factory);
extern void registerSimulatedStateObservationSource(sofa::core::ObjectFactory* factory);
} // namespace container

namespace stochastic
{
#ifdef OPTIMUS_STOCHASTIC_FILTERING
extern void registerFilteringAnimationLoop(sofa::core::ObjectFactory* factory);
extern void registerPreStochasticWrapper(sofa::core::ObjectFactory* factory);
extern void registerStochasticStateWrapper(sofa::core::ObjectFactory* factory);
extern void registerROUKFilter(sofa::core::ObjectFactory* factory);
extern void registerMappedStateObservationManager(sofa::core::ObjectFactory* factory);
#endif
} // namespace stochastic

extern "C" {
    SOFA_OPTIMUSPLUGIN_API void initExternalModule();
    SOFA_OPTIMUSPLUGIN_API const char* getModuleName();
    SOFA_OPTIMUSPLUGIN_API const char* getModuleVersion();
    SOFA_OPTIMUSPLUGIN_API const char* getModuleLicense();
    SOFA_OPTIMUSPLUGIN_API const char* getModuleDescription();
    SOFA_OPTIMUSPLUGIN_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(OPTIMUS_MODULE_NAME);
        first = false;
    }
}

const char* getModuleName()
{
    return OPTIMUS_MODULE_NAME;
}

const char* getModuleVersion()
{
    return OPTIMUS_MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "GPL";
}

const char* getModuleDescription()
{
    return "Bayesian Filtering is a probabilistic technique for data fusion. The technique combines "
           "a concise mathematical formulation of a system with observations of that system. "
           "Probabilities are used to represent the state of a system, and likelihood functions to "
           "represent their relationships";
}

// Called by the ObjectFactory when the PluginManager loads the library (SOFA >= 24.12).
void registerObjects(sofa::core::ObjectFactory* factory)
{
    container::registerOptimParams(factory);
    container::registerSimulatedStateObservationSource(factory);
#ifdef OPTIMUS_STOCHASTIC_FILTERING
    stochastic::registerFilteringAnimationLoop(factory);
    stochastic::registerPreStochasticWrapper(factory);
    stochastic::registerStochasticStateWrapper(factory);
    stochastic::registerROUKFilter(factory);
    stochastic::registerMappedStateObservationManager(factory);
#endif
}

} // namespace component

} // namespace sofa
