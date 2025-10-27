#pragma once

// Windows/MSVC compatibility shim
// Force-included for all host compilations (via /FI or -include)

// ============================================================================
// Prevent Windows SDK pollution
// ============================================================================

#ifdef _WIN32

// Exclude rarely-used Windows APIs to minimize macro pollution
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Prevent min/max macros from <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif

// Prevent COM/RPC headers from being included (source of 'std' macro)
#ifndef RPC_NO_WINDOWS_H
#define RPC_NO_WINDOWS_H
#endif

// If std is already defined as a macro, remove it immediately
#ifdef std
#undef std
#endif

#endif // _WIN32