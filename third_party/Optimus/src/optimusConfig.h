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
#pragma once

#include <sofa/config.h>

// SOFA_BUILD_OPTIMUSPLUGIN is defined by CMake (PRIVATE) while compiling the library.
// SOFA_TARGET is how the ObjectFactory attributes the registered components to this
// plugin (RequiredPlugin checks it); SOFA's own modules define it the same way.
#ifdef SOFA_BUILD_OPTIMUSPLUGIN
#define SOFA_TARGET Optimus
#define SOFA_OPTIMUSPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_OPTIMUSPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#define SOFA_STOCHASTIC_API SOFA_OPTIMUSPLUGIN_API

#define OPTIMUS_MODULE_NAME "Optimus"
#define OPTIMUS_MODULE_VERSION "1.1"
