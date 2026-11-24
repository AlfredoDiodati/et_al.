#include "../../frame/npz.h"
#include "../../frame/sql.h"
#include "../../frame/join.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* --- fixtures produced by real numpy (2.3.3) and embedded here as bytes,
   the same "an external tool validates the format once during
   development, the test embeds the fixed result" pattern
   tests/correctness/gzip_inflate.c uses for its fixed-Huffman block and
   test_npy.c for numpy.load() output - nothing in the shipped suite runs
   Python. Each archive holds three arrays written by np.savez /
   np.savez_compressed:

       vec    a 1D array of four values, this build's dtype
       grid   a 4 x 2 array, exercising the 2D-member-to-columns path
       label  a '<U8' array whose four elements are 1-byte ASCII, an
              empty string, and a mix of 2-, 3- and 4-byte UTF-8
              sequences

   The stored and deflated pairs are the same data through np.savez and
   np.savez_compressed, so the stored/deflate branch of the archive
   reader is the only thing that differs between them. Both precisions
   are embedded because a float32 build refuses a float64 member; a
   float64 build reads both pairs, the float32 one widened.
   tests/correctness/npz_python_interop.py regenerates and re-checks all
   four against a live numpy. --- */

static const unsigned char numpy_stored_f32[918] = {
    0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x00, 0xE2, 0x16, 0x0E, 0x9B, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0x07, 0x00, 0x14, 0x00, 0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70,
    0x79, 0x01, 0x00, 0x10, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x4E, 0x55,
    0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65, 0x73,
    0x63, 0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x66, 0x34, 0x27, 0x2C, 0x20,
    0x27, 0x66, 0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72, 0x64,
    0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65, 0x2C, 0x20,
    0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x34, 0x2C,
    0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x0A, 0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x10,
    0xC0, 0x00, 0x00, 0x96, 0x43, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4B, 0x03,
    0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x85,
    0xFA, 0x94, 0x7D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08,
    0x00, 0x14, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70, 0x79, 0x01,
    0x00, 0x10, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA0,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x4E, 0x55, 0x4D, 0x50,
    0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65, 0x73, 0x63, 0x72,
    0x27, 0x3A, 0x20, 0x27, 0x3C, 0x66, 0x34, 0x27, 0x2C, 0x20, 0x27, 0x66,
    0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72, 0x64, 0x65, 0x72,
    0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65, 0x2C, 0x20, 0x27, 0x73,
    0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x34, 0x2C, 0x20, 0x32,
    0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x0A, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00,
    0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00,
    0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41, 0x00, 0x00, 0x88, 0x41, 0x50,
    0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21,
    0x00, 0x2D, 0x9F, 0x78, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x09, 0x00, 0x14, 0x00, 0x6C, 0x61, 0x62, 0x65, 0x6C, 0x2E, 0x6E,
    0x70, 0x79, 0x01, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x4E,
    0x55, 0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65,
    0x73, 0x63, 0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x55, 0x38, 0x27, 0x2C,
    0x20, 0x27, 0x66, 0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72,
    0x64, 0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65, 0x2C,
    0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x34,
    0x2C, 0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x0A, 0x61, 0x00, 0x00, 0x00, 0x6C, 0x00,
    0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x61, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x61, 0x00,
    0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0xE8, 0x00,
    0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x2D, 0x4E, 0x00, 0x00, 0x00, 0xF6,
    0x01, 0x00, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0xE2, 0x16, 0x0E, 0x9B, 0x90, 0x00,
    0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D,
    0x03, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x85,
    0xFA, 0x94, 0x7D, 0xA0, 0x00, 0x00, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x08,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
    0x01, 0xC9, 0x00, 0x00, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70,
    0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x21, 0x00, 0x2D, 0x9F, 0x78, 0x0F, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0xA3, 0x01, 0x00, 0x00, 0x6C,
    0x61, 0x62, 0x65, 0x6C, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x05, 0x06,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0xA2, 0x00, 0x00, 0x00,
    0xDE, 0x02, 0x00, 0x00, 0x00, 0x00
};

static const unsigned char numpy_deflated_f32[638] = {
    0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x21, 0x00, 0xE2, 0x16, 0x0E, 0x9B, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0x07, 0x00, 0x14, 0x00, 0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70,
    0x79, 0x01, 0x00, 0x10, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x53, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9B, 0xEC, 0x17,
    0xEA, 0x1B, 0x10, 0xC9, 0xC8, 0x50, 0xC6, 0x50, 0xAD, 0x9E, 0x92, 0x5A,
    0x9C, 0x5C, 0xA4, 0x6E, 0xA5, 0xA0, 0x6E, 0x93, 0x66, 0xA2, 0xAE, 0xA3,
    0xA0, 0x9E, 0x96, 0x5F, 0x54, 0x52, 0x94, 0x98, 0x17, 0x9F, 0x5F, 0x94,
    0x92, 0x0A, 0x12, 0x77, 0x4B, 0xCC, 0x29, 0x4E, 0x05, 0x8A, 0x17, 0x67,
    0x24, 0x16, 0xA4, 0x02, 0xF9, 0x1A, 0x26, 0x3A, 0x9A, 0x3A, 0x0A, 0xB5,
    0x0A, 0x14, 0x00, 0x2E, 0x06, 0x86, 0x03, 0xF6, 0x0C, 0x0C, 0x02, 0x07,
    0x18, 0x18, 0xA6, 0x39, 0x33, 0x00, 0x01, 0x00, 0x50, 0x4B, 0x03, 0x04,
    0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x85, 0xFA,
    0x94, 0x7D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08, 0x00,
    0x14, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70, 0x79, 0x01, 0x00,
    0x10, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5D, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9B, 0xEC, 0x17, 0xEA, 0x1B, 0x10,
    0xC9, 0xC8, 0x50, 0xC6, 0x50, 0xAD, 0x9E, 0x92, 0x5A, 0x9C, 0x5C, 0xA4,
    0x6E, 0xA5, 0xA0, 0x6E, 0x93, 0x66, 0xA2, 0xAE, 0xA3, 0xA0, 0x9E, 0x96,
    0x5F, 0x54, 0x52, 0x94, 0x98, 0x17, 0x9F, 0x5F, 0x94, 0x92, 0x0A, 0x12,
    0x77, 0x4B, 0xCC, 0x29, 0x4E, 0x05, 0x8A, 0x17, 0x67, 0x24, 0x16, 0xA4,
    0x02, 0xF9, 0x1A, 0x26, 0x3A, 0x0A, 0x46, 0x9A, 0x3A, 0x0A, 0xB5, 0x0A,
    0x64, 0x03, 0x2E, 0x06, 0x06, 0x05, 0x47, 0x06, 0x06, 0x03, 0x20, 0x76,
    0x00, 0xE2, 0x00, 0x20, 0x4E, 0x00, 0xE2, 0x02, 0x20, 0x6E, 0x00, 0xE2,
    0x0E, 0x47, 0x00, 0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x08,
    0x00, 0x00, 0x00, 0x21, 0x00, 0x2D, 0x9F, 0x78, 0x0F, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x09, 0x00, 0x14, 0x00, 0x6C, 0x61, 0x62,
    0x65, 0x6C, 0x2E, 0x6E, 0x70, 0x79, 0x01, 0x00, 0x10, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x9B, 0xEC, 0x17, 0xEA, 0x1B, 0x10, 0xC9, 0xC8, 0x50, 0xC6,
    0x50, 0xAD, 0x9E, 0x92, 0x5A, 0x9C, 0x5C, 0xA4, 0x6E, 0xA5, 0xA0, 0x6E,
    0x13, 0x6A, 0xA1, 0xAE, 0xA3, 0xA0, 0x9E, 0x96, 0x5F, 0x54, 0x52, 0x94,
    0x98, 0x17, 0x9F, 0x5F, 0x94, 0x92, 0x0A, 0x12, 0x77, 0x4B, 0xCC, 0x29,
    0x4E, 0x05, 0x8A, 0x17, 0x67, 0x24, 0x16, 0xA4, 0x02, 0xF9, 0x1A, 0x26,
    0x3A, 0x9A, 0x3A, 0x0A, 0xB5, 0x0A, 0x14, 0x00, 0xAE, 0x44, 0x06, 0x06,
    0x86, 0x1C, 0x20, 0x2E, 0x00, 0xE2, 0x0C, 0x20, 0x06, 0xF1, 0x91, 0x41,
    0x12, 0x03, 0x65, 0x20, 0x19, 0x6A, 0x66, 0x1A, 0x14, 0xBF, 0x00, 0x62,
    0x05, 0x20, 0xD6, 0xF5, 0x03, 0x12, 0xDF, 0x18, 0x19, 0x00, 0x50, 0x4B,
    0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x21, 0x00, 0xE2, 0x16, 0x0E, 0x9B, 0x53, 0x00, 0x00, 0x00, 0x90, 0x00,
    0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x76, 0x65, 0x63, 0x2E,
    0x6E, 0x70, 0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x85, 0xFA, 0x94, 0x7D, 0x5D,
    0x00, 0x00, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x8C, 0x00, 0x00,
    0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x01,
    0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21,
    0x00, 0x2D, 0x9F, 0x78, 0x0F, 0x68, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x80, 0x01, 0x23, 0x01, 0x00, 0x00, 0x6C, 0x61, 0x62, 0x65, 0x6C,
    0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x03, 0x00, 0xA2, 0x00, 0x00, 0x00, 0xC6, 0x01, 0x00, 0x00,
    0x00, 0x00
};

#ifdef MAT_DOUBLE
static const unsigned char numpy_stored_f64[966] = {
    0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x7F, 0x61, 0xDE, 0x78, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0x07, 0x00, 0x14, 0x00, 0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70,
    0x79, 0x01, 0x00, 0x10, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x4E, 0x55,
    0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65, 0x73,
    0x63, 0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x66, 0x38, 0x27, 0x2C, 0x20,
    0x27, 0x66, 0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72, 0x64,
    0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65, 0x2C, 0x20,
    0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x34, 0x2C,
    0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8,
    0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xC0, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xC0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x21, 0x00, 0x49, 0x4F, 0x7B, 0x8E, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0x08, 0x00, 0x14, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E,
    0x6E, 0x70, 0x79, 0x01, 0x00, 0x10, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93,
    0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64,
    0x65, 0x73, 0x63, 0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x66, 0x38, 0x27,
    0x2C, 0x20, 0x27, 0x66, 0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F,
    0x72, 0x64, 0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65,
    0x2C, 0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28,
    0x34, 0x2C, 0x20, 0x32, 0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x26, 0x40, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x2A, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x40, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x2E, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x30, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x31, 0x40, 0x50,
    0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21,
    0x00, 0x2D, 0x9F, 0x78, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x09, 0x00, 0x14, 0x00, 0x6C, 0x61, 0x62, 0x65, 0x6C, 0x2E, 0x6E,
    0x70, 0x79, 0x01, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x4E,
    0x55, 0x4D, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7B, 0x27, 0x64, 0x65,
    0x73, 0x63, 0x72, 0x27, 0x3A, 0x20, 0x27, 0x3C, 0x55, 0x38, 0x27, 0x2C,
    0x20, 0x27, 0x66, 0x6F, 0x72, 0x74, 0x72, 0x61, 0x6E, 0x5F, 0x6F, 0x72,
    0x64, 0x65, 0x72, 0x27, 0x3A, 0x20, 0x46, 0x61, 0x6C, 0x73, 0x65, 0x2C,
    0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3A, 0x20, 0x28, 0x34,
    0x2C, 0x29, 0x2C, 0x20, 0x7D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x0A, 0x61, 0x00, 0x00, 0x00, 0x6C, 0x00,
    0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x61, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x61, 0x00,
    0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0xE8, 0x00,
    0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x2D, 0x4E, 0x00, 0x00, 0x00, 0xF6,
    0x01, 0x00, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x7F, 0x61, 0xDE, 0x78, 0xA0, 0x00,
    0x00, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D,
    0x03, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x49,
    0x4F, 0x7B, 0x8E, 0xC0, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x08,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
    0x01, 0xD9, 0x00, 0x00, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70,
    0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x21, 0x00, 0x2D, 0x9F, 0x78, 0x0F, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0xD3, 0x01, 0x00, 0x00, 0x6C,
    0x61, 0x62, 0x65, 0x6C, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x05, 0x06,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0xA2, 0x00, 0x00, 0x00,
    0x0E, 0x03, 0x00, 0x00, 0x00, 0x00
};

static const unsigned char numpy_deflated_f64[640] = {
    0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x7F, 0x61, 0xDE, 0x78, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0x07, 0x00, 0x14, 0x00, 0x76, 0x65, 0x63, 0x2E, 0x6E, 0x70,
    0x79, 0x01, 0x00, 0x10, 0x00, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9B, 0xEC, 0x17,
    0xEA, 0x1B, 0x10, 0xC9, 0xC8, 0x50, 0xC6, 0x50, 0xAD, 0x9E, 0x92, 0x5A,
    0x9C, 0x5C, 0xA4, 0x6E, 0xA5, 0xA0, 0x6E, 0x93, 0x66, 0xA1, 0xAE, 0xA3,
    0xA0, 0x9E, 0x96, 0x5F, 0x54, 0x52, 0x94, 0x98, 0x17, 0x9F, 0x5F, 0x94,
    0x92, 0x0A, 0x12, 0x77, 0x4B, 0xCC, 0x29, 0x4E, 0x05, 0x8A, 0x17, 0x67,
    0x24, 0x16, 0xA4, 0x02, 0xF9, 0x1A, 0x26, 0x3A, 0x9A, 0x3A, 0x0A, 0xB5,
    0x0A, 0x14, 0x00, 0x2E, 0x06, 0x30, 0xF8, 0x61, 0x0F, 0xA1, 0x99, 0x0E,
    0x80, 0xA9, 0x03, 0x45, 0x0E, 0x0C, 0x50, 0x00, 0x00, 0x50, 0x4B, 0x03,
    0x04, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x49,
    0x4F, 0x7B, 0x8E, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08,
    0x00, 0x14, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70, 0x79, 0x01,
    0x00, 0x10, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5E,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9B, 0xEC, 0x17, 0xEA, 0x1B,
    0x10, 0xC9, 0xC8, 0x50, 0xC6, 0x50, 0xAD, 0x9E, 0x92, 0x5A, 0x9C, 0x5C,
    0xA4, 0x6E, 0xA5, 0xA0, 0x6E, 0x93, 0x66, 0xA1, 0xAE, 0xA3, 0xA0, 0x9E,
    0x96, 0x5F, 0x54, 0x52, 0x94, 0x98, 0x17, 0x9F, 0x5F, 0x94, 0x92, 0x0A,
    0x12, 0x77, 0x4B, 0xCC, 0x29, 0x4E, 0x05, 0x8A, 0x17, 0x67, 0x24, 0x16,
    0xA4, 0x02, 0xF9, 0x1A, 0x26, 0x3A, 0x0A, 0x46, 0x9A, 0x3A, 0x0A, 0xB5,
    0x0A, 0x64, 0x03, 0x2E, 0x06, 0x30, 0x50, 0x71, 0x80, 0xD0, 0x6A, 0x50,
    0x5A, 0x03, 0x4A, 0x6B, 0x41, 0x69, 0x1D, 0x28, 0xAD, 0x07, 0xA5, 0x0D,
    0xA0, 0xB4, 0xA1, 0x03, 0x00, 0x50, 0x4B, 0x03, 0x04, 0x2D, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x2D, 0x9F, 0x78, 0x0F, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x09, 0x00, 0x14, 0x00, 0x6C,
    0x61, 0x62, 0x65, 0x6C, 0x2E, 0x6E, 0x70, 0x79, 0x01, 0x00, 0x10, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x9B, 0xEC, 0x17, 0xEA, 0x1B, 0x10, 0xC9, 0xC8,
    0x50, 0xC6, 0x50, 0xAD, 0x9E, 0x92, 0x5A, 0x9C, 0x5C, 0xA4, 0x6E, 0xA5,
    0xA0, 0x6E, 0x13, 0x6A, 0xA1, 0xAE, 0xA3, 0xA0, 0x9E, 0x96, 0x5F, 0x54,
    0x52, 0x94, 0x98, 0x17, 0x9F, 0x5F, 0x94, 0x92, 0x0A, 0x12, 0x77, 0x4B,
    0xCC, 0x29, 0x4E, 0x05, 0x8A, 0x17, 0x67, 0x24, 0x16, 0xA4, 0x02, 0xF9,
    0x1A, 0x26, 0x3A, 0x9A, 0x3A, 0x0A, 0xB5, 0x0A, 0x14, 0x00, 0xAE, 0x44,
    0x06, 0x06, 0x86, 0x1C, 0x20, 0x2E, 0x00, 0xE2, 0x0C, 0x20, 0x06, 0xF1,
    0x91, 0x41, 0x12, 0x03, 0x65, 0x20, 0x19, 0x6A, 0x66, 0x1A, 0x14, 0xBF,
    0x00, 0x62, 0x05, 0x20, 0xD6, 0xF5, 0x03, 0x12, 0xDF, 0x18, 0x19, 0x00,
    0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x21, 0x00, 0x7F, 0x61, 0xDE, 0x78, 0x54, 0x00, 0x00, 0x00,
    0xA0, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x76, 0x65,
    0x63, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D,
    0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0x49, 0x4F, 0x7B,
    0x8E, 0x5E, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x8D,
    0x00, 0x00, 0x00, 0x67, 0x72, 0x69, 0x64, 0x2E, 0x6E, 0x70, 0x79, 0x50,
    0x4B, 0x01, 0x02, 0x2D, 0x03, 0x2D, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00,
    0x00, 0x21, 0x00, 0x2D, 0x9F, 0x78, 0x0F, 0x68, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x80, 0x01, 0x25, 0x01, 0x00, 0x00, 0x6C, 0x61, 0x62,
    0x65, 0x6C, 0x2E, 0x6E, 0x70, 0x79, 0x50, 0x4B, 0x05, 0x06, 0x00, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0xA2, 0x00, 0x00, 0x00, 0xC8, 0x01,
    0x00, 0x00, 0x00, 0x00
};
#endif


static const char *utf8_fixture_labels[4] = { "alpha", "b", "", "caff\xc3\xa8 \xe4\xb8\xad\xf0\x9f\x98\x80" };

static void write_bytes(const char *path, const unsigned char *bytes, size_t n) {
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite(bytes, 1, n, f);
    fclose(f);
}

/* --- a test-only .npz builder, deliberately a second implementation of
   the format rather than a call into df_write_npz: it is what lets a
   malformed or unsupported archive be constructed on purpose, and what
   the reader is checked against. Same relationship test_npy.c's
   write_test_npy has to df_write_npy. --- */

typedef struct { const char *key; const char *descr; const void *data; size_t data_bytes; int rows; int cols; } FixtureArray;

static size_t build_npy_image(const FixtureArray *a, unsigned char *out) {
    char shape[64];
    if (a->cols == 1) snprintf(shape, sizeof shape, "(%d,)", a->rows);
    else snprintf(shape, sizeof shape, "(%d, %d)", a->rows, a->cols);
    char content[256];
    int content_len = snprintf(content, sizeof content,
        "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", a->descr, shape);
    int pad = (64 - ((10 + content_len + 1) % 64)) % 64;
    int header_len = content_len + pad + 1;

    memcpy(out, "\x93NUMPY", 6);
    out[6] = 1; out[7] = 0;
    out[8] = (unsigned char)(header_len & 0xFF);
    out[9] = (unsigned char)((header_len >> 8) & 0xFF);
    memcpy(out + 10, content, (size_t)content_len);
    memset(out + 10 + content_len, ' ', (size_t)pad);
    out[10 + header_len - 1] = '\n';
    memcpy(out + 10 + header_len, a->data, a->data_bytes);
    return (size_t)(10 + header_len) + a->data_bytes;
}

static void put16(unsigned char *p, unsigned v) { p[0] = (unsigned char)(v & 0xFF); p[1] = (unsigned char)(v >> 8); }
static void put32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFF); p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF); p[3] = (unsigned char)((v >> 24) & 0xFF);
}

static void build_fixture_npz(const char *path, const FixtureArray *arrays, int n) {
    unsigned char *images[8];
    size_t lens[8], offsets[8];
    uint32_t crcs[8];
    assert(n <= 8);

    FILE *f = fopen(path, "wb");
    assert(f);
    long offset = 0;
    for (int i = 0; i < n; i++) {
        images[i] = (unsigned char*)malloc(1024 + arrays[i].data_bytes);
        lens[i] = build_npy_image(&arrays[i], images[i]);
        crcs[i] = gzip_crc32(images[i], lens[i]);
        offsets[i] = (size_t)offset;

        char name[128];
        int name_len = snprintf(name, sizeof name, "%s.npy", arrays[i].key);
        unsigned char lh[30];
        memset(lh, 0, sizeof lh);
        put32(lh, 0x04034b50u);
        put16(lh + 4, 20);
        put32(lh + 14, crcs[i]);
        put32(lh + 18, (uint32_t)lens[i]);
        put32(lh + 22, (uint32_t)lens[i]);
        put16(lh + 26, (unsigned)name_len);
        fwrite(lh, 1, sizeof lh, f);
        fwrite(name, 1, (size_t)name_len, f);
        fwrite(images[i], 1, lens[i], f);
        offset += (long)sizeof lh + name_len + (long)lens[i];
    }

    long cd_offset = offset;
    for (int i = 0; i < n; i++) {
        char name[128];
        int name_len = snprintf(name, sizeof name, "%s.npy", arrays[i].key);
        unsigned char ch[46];
        memset(ch, 0, sizeof ch);
        put32(ch, 0x02014b50u);
        put16(ch + 4, 20);
        put16(ch + 6, 20);
        put32(ch + 16, crcs[i]);
        put32(ch + 20, (uint32_t)lens[i]);
        put32(ch + 24, (uint32_t)lens[i]);
        put16(ch + 28, (unsigned)name_len);
        put32(ch + 42, (uint32_t)offsets[i]);
        fwrite(ch, 1, sizeof ch, f);
        fwrite(name, 1, (size_t)name_len, f);
        offset += (long)sizeof ch + name_len;
        free(images[i]);
    }

    unsigned char eocd[22];
    memset(eocd, 0, sizeof eocd);
    put32(eocd, 0x06054b50u);
    put16(eocd + 8, (unsigned)n);
    put16(eocd + 10, (unsigned)n);
    put32(eocd + 12, (uint32_t)(offset - cd_offset));
    put32(eocd + 16, (uint32_t)cd_offset);
    fwrite(eocd, 1, sizeof eocd, f);
    fclose(f);
}

/* --- the checks --- */

static void test_numeric_roundtrip(void) {
    puts("numeric round-trip: values, column names and order survive write/read");

    const char *path = "out/test_npz_numeric.npz";
    DataFrame df = df_new(3);
    Vec gdp = mat_lit(3, 1, 1.5f, -2.25f, 300.f);
    Vec cpi = mat_lit(3, 1, 0.f, 1e-30f, 1e30f);
    df_add_numeric_col(&df, "gdp", gdp);
    df_add_numeric_col(&df, "cpi", cpi);

    df_write_npz(&df, path);
    DataFrame back = df_read_npz(path);

    assert(back.r == 3 && back.n_cols == 2 && back.n_string == 0);
    assert(strcmp(back.columns[0].name, "gdp") == 0);
    assert(strcmp(back.columns[1].name, "cpi") == 0);
    /* exact, not toleranced: .npz carries raw bytes, so any difference is
       a byte-layout bug rather than a formatting-precision question */
    for (int i = 0; i < 3; i++) {
        assert(AT(back.numeric, i, 0) == AT(df.numeric, i, 0));
        assert(AT(back.numeric, i, 1) == AT(df.numeric, i, 1));
    }
    assert(back.row_names == NULL);

    mat_free(gdp); mat_free(cpi);
    df_free(&df); df_free(&back);
    remove(path);
}

static void test_mixed_columns_preserve_declaration_order(void) {
    puts("mixed numeric/string columns keep declaration order through the archive");

    const char *path = "out/test_npz_mixed.npz";
    DataFrame df = df_new(3);
    const char *quarter[3] = { "1973 March", "1973 June", "1973 September" };
    Vec gdp = mat_lit(3, 1, 5642.669f, 5704.098f, 5748.5f);
    const char *region[3] = { "north", "south", "east" };

    df_add_string_col(&df, "quarter", quarter);
    df_add_numeric_col(&df, "gdp", gdp);
    df_add_string_col(&df, "region", region);

    df_write_npz(&df, path);
    DataFrame back = df_read_npz(path);

    assert(back.n_cols == 3 && back.n_string == 2);
    assert(back.columns[0].type == COL_STRING && strcmp(back.columns[0].name, "quarter") == 0);
    assert(back.columns[1].type == COL_NUMERIC && strcmp(back.columns[1].name, "gdp") == 0);
    assert(back.columns[2].type == COL_STRING && strcmp(back.columns[2].name, "region") == 0);
    for (int i = 0; i < 3; i++) {
        assert(strcmp(df_col_string(&back, "quarter")[i], quarter[i]) == 0);
        assert(strcmp(df_col_string(&back, "region")[i], region[i]) == 0);
        CHECK(AT(df_col_numeric(&back, "gdp"), i, 0), AT(gdp, i, 0));
    }

    mat_free(gdp);
    df_free(&df); df_free(&back);
    remove(path);
}

static void test_row_names_roundtrip(void) {
    puts("row labels survive as the reserved " FRAME_NPZ_ROW_NAMES_KEY " member, not as a column");

    const char *path = "out/test_npz_rownames.npz";
    DataFrame df = df_new(3);
    Vec x = mat_lit(3, 1, 1.f, 2.f, 3.f);
    df_add_numeric_col(&df, "x", x);
    const char *names[3] = { "r1", "r2", "r3" };
    df_set_row_names(&df, names);

    df_write_npz(&df, path);
    DataFrame back = df_read_npz(path);

    assert(back.n_cols == 1 && back.n_string == 0);
    assert(back.row_names != NULL);
    for (int i = 0; i < 3; i++) assert(strcmp(back.row_names[i], names[i]) == 0);

    /* a frame without row labels must come back without them, so the
       reserved member is genuinely absent rather than written empty */
    DataFrame plain = df_new(3);
    df_add_numeric_col(&plain, "x", x);
    df_write_npz(&plain, path);
    DataFrame plain_back = df_read_npz(path);
    assert(plain_back.row_names == NULL && plain_back.n_cols == 1);

    mat_free(x);
    df_free(&df); df_free(&back); df_free(&plain); df_free(&plain_back);
    remove(path);
}

static void test_unicode_strings_roundtrip(void) {
    puts("UTF-8 string columns round-trip through numpy's UCS-4 dtype (1-, 2-, 3- and 4-byte sequences)");

    const char *path = "out/test_npz_unicode.npz";
    DataFrame df = df_new(4);
    Vec x = mat_lit(4, 1, 1.f, 2.f, 3.f, 4.f);
    df_add_numeric_col(&df, "x", x);
    df_add_string_col(&df, "label", utf8_fixture_labels);

    df_write_npz(&df, path);
    DataFrame back = df_read_npz(path);

    char **got = df_col_string(&back, "label");
    for (int i = 0; i < 4; i++) assert(strcmp(got[i], utf8_fixture_labels[i]) == 0);

    mat_free(x);
    df_free(&df); df_free(&back);
    remove(path);
}

/* The fixture's `grid` member is a genuine 4 x 2 array, which nothing
   df_write_npz produces (it writes one 1D member per column) - the only
   way this branch runs is a file numpy wrote. */
static void check_numpy_fixture(const char *label, const unsigned char *bytes, size_t n) {
    printf("real numpy archive (%s): values, dtypes, 2D expansion and unicode all read back\n", label);

    const char *path = "out/test_npz_numpy.npz";
    write_bytes(path, bytes, n);
    DataFrame df = df_read_npz(path);

    assert(df.r == 4);
    assert(df.n_cols == 4 && df.n_string == 1); /* vec, grid0, grid1, label */
    assert(strcmp(df.columns[0].name, "vec") == 0);
    assert(strcmp(df.columns[1].name, "grid0") == 0);
    assert(strcmp(df.columns[2].name, "grid1") == 0);
    assert(strcmp(df.columns[3].name, "label") == 0);

    Mat vec = df_col_numeric(&df, "vec");
    CHECK(AT(vec, 0, 0), 1.5f); CHECK(AT(vec, 1, 0), -2.25f);
    CHECK(AT(vec, 2, 0), 300.f); CHECK(AT(vec, 3, 0), 0.f);

    Mat g0 = df_col_numeric(&df, "grid0"), g1 = df_col_numeric(&df, "grid1");
    for (int i = 0; i < 4; i++) {
        CHECK(AT(g0, i, 0), (mreal)(10 + 2 * i));
        CHECK(AT(g1, i, 0), (mreal)(11 + 2 * i));
    }

    char **labels = df_col_string(&df, "label");
    for (int i = 0; i < 4; i++) assert(strcmp(labels[i], utf8_fixture_labels[i]) == 0);

    df_free(&df);
    remove(path);
}

static void test_reads_real_numpy_archives(void) {
    check_numpy_fixture("np.savez float32, stored", numpy_stored_f32, sizeof numpy_stored_f32);
    check_numpy_fixture("np.savez_compressed float32, deflated", numpy_deflated_f32, sizeof numpy_deflated_f32);
#ifdef MAT_DOUBLE
    check_numpy_fixture("np.savez float64, stored", numpy_stored_f64, sizeof numpy_stored_f64);
    check_numpy_fixture("np.savez_compressed float64, deflated", numpy_deflated_f64, sizeof numpy_deflated_f64);
#endif
}

static void test_adversarial_shapes(void) {
    puts("adversarial: one row, one column, and an all-empty string column");

    const char *path = "out/test_npz_adversarial.npz";
    DataFrame single = df_new(1);
    Vec one = mat_lit(1, 1, 99.f);
    df_add_numeric_col(&single, "only", one);
    df_write_npz(&single, path);
    DataFrame single_back = df_read_npz(path);
    assert(single_back.r == 1 && single_back.n_cols == 1);
    CHECK(AT(single_back.numeric, 0, 0), 99.f);

    /* every element empty: numpy never emits a '<U0', so the writer has
       to widen to one codepoint rather than emit a zero-width dtype */
    DataFrame empties = df_new(2);
    const char *blank[2] = { "", "" };
    df_add_string_col(&empties, "blank", blank);
    df_write_npz(&empties, path);
    DataFrame empties_back = df_read_npz(path);
    assert(empties_back.n_string == 1);
    assert(strcmp(df_col_string(&empties_back, "blank")[0], "") == 0);
    assert(strcmp(df_col_string(&empties_back, "blank")[1], "") == 0);

    /* zero rows. A legitimate numpy array (np.savez of an empty one produces
       exactly this), not a contract violation, and the only shape where a
       member's data block is empty and the .npy image is header alone. */
    DataFrame no_rows = df_new(0);
    Vec none = mat_new(0, 1);
    df_add_numeric_col(&no_rows, "x", none);
    df_add_string_col(&no_rows, "s", NULL);
    df_write_npz(&no_rows, path);
    DataFrame no_rows_back = df_read_npz(path);
    assert(no_rows_back.r == 0 && no_rows_back.n_cols == 2 && no_rows_back.n_string == 1);
    assert(strcmp(no_rows_back.columns[0].name, "x") == 0);
    assert(no_rows_back.columns[1].type == COL_STRING);

    mat_free(one); mat_free(none);
    df_free(&single); df_free(&single_back); df_free(&empties); df_free(&empties_back);
    df_free(&no_rows); df_free(&no_rows_back);
    remove(path);
}

static int df_identical(const DataFrame *a, const DataFrame *b);

/* df_write_npz_compressed differs from df_write_npz only in each
   member's payload, so what has to hold is that the archive means the
   same thing: the same frame back, and a genuinely smaller file when the
   data is compressible. The deflated read path is otherwise only
   exercised by the embedded numpy fixture, which is one archive at one
   size; this reaches it at every shape the suite builds. */
static void test_compressed_archive_means_the_same_thing(void) {
    puts("df_write_npz_compressed: smaller file, same frame back");

    const char *plain_path = "out/test_npz_plain.npz";
    const char *compressed_path = "out/test_npz_compressed.npz";

    int r = 2000;
    DataFrame df = df_new(r);
    Vec ramp = mat_new(r, 1);
    for (int i = 0; i < r; i++) ramp.d[i] = (mreal)(i % 64);
    df_add_numeric_col(&df, "ramp", ramp);
    const char **label = (const char**)malloc((size_t)r * sizeof(char*));
    const char *pool[4] = { "north", "south", "east", "west" };
    for (int i = 0; i < r; i++) label[i] = pool[i % 4];
    df_add_string_col(&df, "region", label);
    const char **rows = (const char**)malloc((size_t)r * sizeof(char*));
    for (int i = 0; i < r; i++) rows[i] = pool[(i + 3) % 4];
    df_set_row_names(&df, rows);

    df_write_npz(&df, plain_path);
    df_write_npz_compressed(&df, compressed_path);

    long plain_size, compressed_size;
    free(frame_read_file(plain_path, &plain_size));
    free(frame_read_file(compressed_path, &compressed_size));
    assert(compressed_size < plain_size / 2 &&
           "compressible data should compress to well under half, or the deflate path is not running");

    DataFrame from_plain = df_read_npz(plain_path);
    DataFrame from_compressed = df_read_npz(compressed_path);
    assert(df_identical(&from_plain, &from_compressed));
    assert(from_compressed.row_names != NULL);

    /* incompressible data must not make the archive bigger: a member
       whose deflated form is no smaller is stored instead */
    DataFrame noisy = df_new(r);
    Vec noise = mat_new(r, 1);
    for (int i = 0; i < r; i++) noise.d[i] = (mreal)((unsigned)(i * 2654435761u) % 1000000) / (mreal)7.0;
    df_add_numeric_col(&noisy, "noise", noise);
    df_write_npz(&noisy, plain_path);
    df_write_npz_compressed(&noisy, compressed_path);
    long noisy_plain, noisy_compressed;
    free(frame_read_file(plain_path, &noisy_plain));
    free(frame_read_file(compressed_path, &noisy_compressed));
    assert(noisy_compressed <= noisy_plain &&
           "compressing incompressible data must not grow the archive");
    DataFrame noisy_back = df_read_npz(compressed_path);
    assert(df_identical(&noisy, &noisy_back));

    /* a one-row frame through the compressed path, where a member's
       payload is a few bytes and the block-type choice is not obvious */
    DataFrame tiny = df_new(1);
    Vec single = mat_lit(1, 1, 42.f);
    df_add_numeric_col(&tiny, "x", single);
    df_write_npz_compressed(&tiny, compressed_path);
    DataFrame tiny_back = df_read_npz(compressed_path);
    assert(df_identical(&tiny, &tiny_back));

    free(label); free(rows);
    mat_free(ramp); mat_free(noise); mat_free(single);
    df_free(&df); df_free(&from_plain); df_free(&from_compressed);
    df_free(&noisy); df_free(&noisy_back); df_free(&tiny); df_free(&tiny_back);
    remove(plain_path); remove(compressed_path);
}

/* df_read_npz builds columns[].index itself - which numeric slot a column
   occupies, which string_cols slot, and in what declaration order - rather
   than inheriting it from a frame that already existed. Every other check
   here reads that metadata through df_col_numeric/df_col_string, which is
   one lookup and would still pass if an index were wrong in a way the two
   happened to agree on. df_sql and df_join are the heaviest consumers of
   the whole array: they walk it, group by a string column, and build a new
   frame out of the result. A frame from the archive has to be as good an
   input to them as the frame it came from. */
static void test_rebuilt_metadata_drives_sql_and_join(void) {
    puts("a frame rebuilt from the archive is a valid df_sql and df_join input");

    const char *path = "out/test_npz_metadata.npz";
    DataFrame df = df_new(4);
    const char *region[4] = { "north", "south", "north", "east" };
    Vec gdp = mat_lit(4, 1, 10.f, 20.f, 30.f, 40.f);
    Vec year = mat_lit(4, 1, 2020.f, 2021.f, 2022.f, 2023.f);
    df_add_string_col(&df, "region", region);
    df_add_numeric_col(&df, "gdp", gdp);
    df_add_numeric_col(&df, "year", year);

    df_write_npz(&df, path);
    DataFrame back = df_read_npz(path);

    /* the same query through both frames, compared row by row */
    const char *query = "SELECT region, SUM(gdp) AS total FROM df GROUP BY region ORDER BY region";
    DataFrame from_original = df_sql(&df, query);
    DataFrame from_archive = df_sql(&back, query);
    assert(from_archive.r == from_original.r);
    assert(from_archive.n_cols == from_original.n_cols);
    for (int i = 0; i < from_original.r; i++) {
        assert(strcmp(df_col_string(&from_archive, "region")[i],
                      df_col_string(&from_original, "region")[i]) == 0);
        CHECK(AT(df_col_numeric(&from_archive, "total"), i, 0),
              AT(df_col_numeric(&from_original, "total"), i, 0));
    }
    /* the query has to have done something: three distinct regions out of
       four rows, so a pass-through would show four */
    assert(from_archive.r == 3);

    DataFrame joined = df_join(&back, &back, "year", JOIN_INNER);
    assert(joined.r == 4);
    assert(strcmp(df_col_string(&joined, "region")[0], "north") == 0);

    /* a filter on the numeric column whose index the archive assigned */
    DataFrame filtered = df_sql(&back, "SELECT gdp FROM df WHERE gdp > 15");
    assert(filtered.r == 3);
    CHECK(AT(df_col_numeric(&filtered, "gdp"), 0, 0), 20.f);

    mat_free(gdp); mat_free(year);
    df_free(&df); df_free(&back);
    df_free(&from_original); df_free(&from_archive);
    df_free(&joined); df_free(&filtered);
    remove(path);
}

/* --- rejection paths. Every one is an assert (this project's contract
   convention), so each is exercised in a forked child and required to
   die on SIGABRT - the technique test_npy.c established. --- */

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111); /* fn() must never return */
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static const char *g_bad_path = "out/test_npz_bad.npz";
static void call_df_read_npz(void) { DataFrame df = df_read_npz(g_bad_path); (void)df; }

static void test_malformed_npz_aborts(void) {
    puts("malformed .npz aborts (fork + expect SIGABRT)");

    unsigned char *valid;
    long valid_len;
    {
        DataFrame df = df_new(2);
        Vec x = mat_lit(2, 1, 1.f, 2.f);
        df_add_numeric_col(&df, "x", x);
        df_write_npz(&df, g_bad_path);
        mat_free(x); df_free(&df);
        valid = (unsigned char*)frame_read_file(g_bad_path, &valid_len);
    }

    /* no end-of-central-directory record at all */
    {
        unsigned char junk[64];
        memset(junk, 0x41, sizeof junk);
        write_bytes(g_bad_path, junk, sizeof junk);
        expect_abort(call_df_read_npz);
    }

    /* too small to hold even an EOCD */
    write_bytes(g_bad_path, valid, 8);
    expect_abort(call_df_read_npz);

    /* member data truncated away, EOCD and directory left intact */
    write_bytes(g_bad_path, valid, (size_t)valid_len - 30);
    expect_abort(call_df_read_npz);

    /* a flipped byte inside the first member's own bytes: the CRC32 the
       directory records no longer matches what the member holds. The
       offset comes from the local header rather than being counted back
       from the end of the file, which lands in the directory instead. */
    {
        unsigned char *corrupt = (unsigned char*)malloc((size_t)valid_len);
        memcpy(corrupt, valid, (size_t)valid_len);
        size_t member_data = 30 + frame_npz_u16(corrupt + 26) + frame_npz_u16(corrupt + 28);
        assert(member_data + 130 < (size_t)valid_len);
        corrupt[member_data + 130] ^= 0xFF; /* past the 64-byte-aligned preamble, in the values */
        write_bytes(g_bad_path, corrupt, (size_t)valid_len);
        expect_abort(call_df_read_npz);
        free(corrupt);
    }

    /* central-directory signature broken */
    {
        unsigned char *corrupt = (unsigned char*)malloc((size_t)valid_len);
        memcpy(corrupt, valid, (size_t)valid_len);
        uint32_t cd_offset = frame_npz_u32(corrupt + (size_t)valid_len - 22 + 16);
        corrupt[cd_offset + 3] ^= 0xFF;
        write_bytes(g_bad_path, corrupt, (size_t)valid_len);
        expect_abort(call_df_read_npz);
        free(corrupt);
    }

    free(valid);

    /* a dtype that is neither this build's mreal nor a '<U' string */
    {
        int32_t ints[3] = { 1, 2, 3 };
        FixtureArray a = { "counts", "<i4", ints, sizeof ints, 3, 1 };
        build_fixture_npz(g_bad_path, &a, 1);
        expect_abort(call_df_read_npz);
    }

    /* two members disagreeing about the row count */
    {
        mreal two[2] = { 1, 2 }, three[3] = { 1, 2, 3 };
        FixtureArray a[2] = {
            { "short", frame_npy_mreal_descr(), two, sizeof two, 2, 1 },
            { "long", frame_npy_mreal_descr(), three, sizeof three, 3, 1 }
        };
        build_fixture_npz(g_bad_path, a, 2);
        expect_abort(call_df_read_npz);
    }

    /* an archive with no arrays in it at all */
    {
        FixtureArray none = { NULL, NULL, NULL, 0, 0, 0 };
        (void)none;
        build_fixture_npz(g_bad_path, NULL, 0);
        expect_abort(call_df_read_npz);
    }

#ifndef MAT_DOUBLE
    /* a float64 member in a float32 build, which would round: the same
       contract a bare .npy read enforces. A float64 build reads a float32
       member, which tests/correctness/float32_file_widening.c checks. */
    {
        unsigned char payload[64];
        memset(payload, 0, sizeof payload);
        FixtureArray a = { "x", "<f8", payload, 24, 3, 1 };
        build_fixture_npz(g_bad_path, &a, 1);
        expect_abort(call_df_read_npz);
    }
#endif

    remove(g_bad_path);
}

static const char *g_write_path = "out/test_npz_write_bad.npz";

static void write_empty_frame(void) {
    DataFrame df = df_new(4);
    df_write_npz(&df, g_write_path);
    df_free(&df);
}

static void write_reserved_column_name(void) {
    DataFrame df = df_new(2);
    Vec x = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&df, FRAME_NPZ_ROW_NAMES_KEY, x);
    df_write_npz(&df, g_write_path);
    mat_free(x); df_free(&df);
}

static void test_write_contract_aborts(void) {
    puts("df_write_npz contract violations abort (empty frame, reserved column name)");
    expect_abort(write_empty_frame);
    expect_abort(write_reserved_column_name);
    remove(g_write_path);
}

/* --- df_read_npz_columns. Its reference is df_read_npz on the same file:
   every column it returns must be the full read's column of that name, bit
   for bit, in the order asked, with the full read's row labels. --- */

static int column_position(const DataFrame *df, const char *name) {
    for (int j = 0; j < df->n_cols; j++)
        if (strcmp(df->columns[j].name, name) == 0) return j;
    return -1;
}

/* Whether column j of a and the column of b named name have the same name,
   type and bits. Numeric values are compared with memcmp rather than ==,
   since the archive carries raw bytes and a reader that changed one is
   wrong whatever the value. */
static int same_column(const DataFrame *a, int j, const DataFrame *b, const char *name) {
    int k = column_position(b, name);
    if (j < 0 || j >= a->n_cols || k < 0 || a->r != b->r) return 0;
    if (strcmp(a->columns[j].name, name) != 0 || a->columns[j].type != b->columns[k].type) return 0;
    if (a->columns[j].type == COL_NUMERIC) {
        for (int i = 0; i < a->r; i++)
            if (memcmp(&AT(a->numeric, i, a->columns[j].index), &AT(b->numeric, i, b->columns[k].index),
                       sizeof(mreal)) != 0) return 0;
        return 1;
    }
    for (int i = 0; i < a->r; i++)
        if (strcmp(a->string_cols[a->columns[j].index][i], b->string_cols[b->columns[k].index][i]) != 0) return 0;
    return 1;
}

/* Whether subset is what df_read_npz_columns(names) should return given
   the full read of the same archive: the names full holds, in the order
   asked, each identical to full's column, and full's row labels; or, when
   full holds none of the names, a frame of no rows and no columns. Only
   for archives whose members are all 1D, where a key is a column name. */
static int matches_full_read(const DataFrame *subset, const DataFrame *full, const char *const *names, int n_names) {
    int j = 0;
    for (int k = 0; k < n_names; k++) {
        if (column_position(full, names[k]) < 0) continue;
        if (!same_column(subset, j, full, names[k])) return 0;
        j++;
    }
    if (subset->n_cols != j) return 0;
    if (j == 0) return subset->r == 0 && subset->row_names == NULL;
    if ((subset->row_names == NULL) != (full->row_names == NULL)) return 0;
    if (subset->row_names)
        for (int i = 0; i < subset->r; i++)
            if (strcmp(subset->row_names[i], full->row_names[i]) != 0) return 0;
    return 1;
}

static void test_column_subset_known_values(void) {
    puts("df_read_npz_columns: asked columns in the asked order, with the row labels");

    const char *path = "out/test_npz_columns_known.npz";
    DataFrame df = df_new(3);
    Vec a = mat_lit(3, 1, 1.5f, -2.f, 0.f);
    Vec b = mat_lit(3, 1, 10.f, 20.f, 30.f);
    Vec c = mat_lit(3, 1, -1e30f, 1e-30f, 7.f);
    const char *s[3] = { "north", "", "caff\xc3\xa8" };
    const char *rows[3] = { "r0", "r1", "r2" };
    df_add_numeric_col(&df, "a", a);
    df_add_string_col(&df, "s", s);
    df_add_numeric_col(&df, "b", b);
    df_add_numeric_col(&df, "c", c);
    df_set_row_names(&df, rows);

    for (int compressed = 0; compressed < 2; compressed++) {
        if (compressed) df_write_npz_compressed(&df, path);
        else df_write_npz(&df, path);

        const char *asked[3] = { "c", "s", "a" };
        DataFrame back = df_read_npz_columns(path, asked, 3);
        assert(back.r == 3 && back.n_cols == 3 && back.n_string == 1);
        assert(strcmp(back.columns[0].name, "c") == 0 && back.columns[0].type == COL_NUMERIC);
        assert(strcmp(back.columns[1].name, "s") == 0 && back.columns[1].type == COL_STRING);
        assert(strcmp(back.columns[2].name, "a") == 0 && back.columns[2].type == COL_NUMERIC);
        Mat back_c = df_col_numeric(&back, "c"), back_a = df_col_numeric(&back, "a");
        for (int i = 0; i < 3; i++) {
            assert(AT(back_c, i, 0) == AT(c, i, 0));
            assert(AT(back_a, i, 0) == AT(a, i, 0));
            assert(strcmp(df_col_string(&back, "s")[i], s[i]) == 0);
            assert(back.row_names && strcmp(back.row_names[i], rows[i]) == 0);
        }
        df_free(&back);

        /* one column, the last member before the row labels */
        const char *only_b[1] = { "b" };
        DataFrame just_b = df_read_npz_columns(path, only_b, 1);
        assert(just_b.n_cols == 1 && just_b.r == 3 && just_b.row_names);
        for (int i = 0; i < 3; i++) assert(AT(just_b.numeric, i, 0) == AT(b, i, 0));
        df_free(&just_b);
    }

    mat_free(a); mat_free(b); mat_free(c);
    df_free(&df);
    remove(path);
}

/* A frame of 1000 rows and 300 numeric columns plus a string column, so
   the archive is larger than the 22 + 65535 bytes the end-of-directory
   search reads from the file's tail. Values from a fixed seed, biased
   toward zeros, huge and tiny magnitudes. */
static DataFrame wide_frame(int with_row_names) {
    srand(47);
    int r = 1000, n_numeric = 300;
    DataFrame df = df_new(r);
    Vec column = mat_new(r, 1);
    char name[32];
    for (int j = 0; j < n_numeric; j++) {
        for (int i = 0; i < r; i++) {
            switch (rand() % 5) {
                case 0: column.d[i] = 0; break;
                case 1: column.d[i] = (mreal)(rand() % 2 ? 1 : -1) * (mreal)1e30; break;
                case 2: column.d[i] = (mreal)1e-30; break;
                default: column.d[i] = (mreal)rand() / (mreal)RAND_MAX - (mreal)0.5; break;
            }
        }
        snprintf(name, sizeof name, "v_%d", j);
        df_add_numeric_col(&df, name, column);
    }
    const char **labels = (const char**)malloc((size_t)r * sizeof(char*));
    char **owned = (char**)malloc((size_t)r * sizeof(char*));
    for (int i = 0; i < r; i++) {
        snprintf(name, sizeof name, "row %d", i);
        owned[i] = frame_strdup(name);
        labels[i] = owned[i];
    }
    df_add_string_col(&df, "label", labels);
    if (with_row_names) df_set_row_names(&df, labels);
    for (int i = 0; i < r; i++) free(owned[i]);
    free(owned);
    free(labels);
    mat_free(column);
    return df;
}

static void test_column_subset_agrees_with_full_read(void) {
    puts("df_read_npz_columns: agrees with df_read_npz on a 1000 x 301 archive, stored and deflated");

    const char *path = "out/test_npz_columns_wide.npz";
    for (int with_row_names = 0; with_row_names < 2; with_row_names++) {
        DataFrame df = wide_frame(with_row_names);
        for (int compressed = 0; compressed < 2; compressed++) {
            if (compressed) df_write_npz_compressed(&df, path);
            else df_write_npz(&df, path);
            DataFrame full = df_read_npz(path);
            assert(df_identical(&df, &full));

            /* out of archive order, the first and last members among them,
               and a name the archive does not hold */
            const char *asked[] = { "v_217", "v_3", "label", "v_0", "v_299", "v_150", "v_42", "v_42b", "v_88",
                                    "v_1", "v_270", "v_9" };
            int n_asked = (int)(sizeof asked / sizeof *asked);
            DataFrame some = df_read_npz_columns(path, asked, n_asked);
            assert(some.n_cols == n_asked - 1 && some.r == 1000);
            assert(column_position(&some, "v_42b") < 0);
            assert(matches_full_read(&some, &full, asked, n_asked));
            df_free(&some);

            /* every column, in archive order and reversed */
            const char **every = (const char**)malloc((size_t)full.n_cols * sizeof(char*));
            for (int k = 0; k < full.n_cols; k++) every[k] = full.columns[k].name;
            DataFrame all = df_read_npz_columns(path, every, full.n_cols);
            assert(df_identical(&all, &full));
            df_free(&all);
            for (int k = 0; k < full.n_cols; k++) every[k] = full.columns[full.n_cols - 1 - k].name;
            DataFrame reversed = df_read_npz_columns(path, every, full.n_cols);
            assert(matches_full_read(&reversed, &full, every, full.n_cols));
            df_free(&reversed);
            free(every);

            /* no name found: a frame of no rows and no columns, and no row
               labels even when the archive has them */
            const char *absent[2] = { "v_300", "v" };
            DataFrame none = df_read_npz_columns(path, absent, 2);
            assert(none.r == 0 && none.n_cols == 0 && none.row_names == NULL);
            df_free(&none);
            df_free(&full);
        }
        df_free(&df);
    }
    remove(path);
}

/* numpy's own archives carry a 2D member, which df_write_npz never writes,
   and ZIP64 extra fields in their local headers. A 2D member is asked for
   by its key and comes back as all of its columns. */
static void check_numpy_fixture_columns(const char *label, const unsigned char *bytes, size_t n) {
    printf("df_read_npz_columns on a real numpy archive (%s)\n", label);
    const char *path = "out/test_npz_columns_numpy.npz";
    write_bytes(path, bytes, n);
    DataFrame full = df_read_npz(path);

    const char *asked[2] = { "label", "grid" };
    DataFrame some = df_read_npz_columns(path, asked, 2);
    assert(some.r == 4 && some.n_cols == 3);
    assert(same_column(&some, 0, &full, "label"));
    assert(same_column(&some, 1, &full, "grid0"));
    assert(same_column(&some, 2, &full, "grid1"));
    df_free(&some);

    /* a generated column name is not a member key */
    const char *generated[1] = { "grid0" };
    DataFrame none = df_read_npz_columns(path, generated, 1);
    assert(none.n_cols == 0 && none.r == 0);
    df_free(&none);

    const char *vec[1] = { "vec" };
    DataFrame just_vec = df_read_npz_columns(path, vec, 1);
    assert(just_vec.n_cols == 1 && same_column(&just_vec, 0, &full, "vec"));
    df_free(&just_vec);

    df_free(&full);
    remove(path);
}

static void test_column_subset_reads_real_numpy_archives(void) {
    check_numpy_fixture_columns("np.savez float32, stored", numpy_stored_f32, sizeof numpy_stored_f32);
    check_numpy_fixture_columns("np.savez_compressed float32, deflated", numpy_deflated_f32, sizeof numpy_deflated_f32);
#ifdef MAT_DOUBLE
    check_numpy_fixture_columns("np.savez float64, stored", numpy_stored_f64, sizeof numpy_stored_f64);
    check_numpy_fixture_columns("np.savez_compressed float64, deflated", numpy_deflated_f64, sizeof numpy_deflated_f64);
#endif
}

static void test_column_subset_adversarial_shapes(void) {
    puts("df_read_npz_columns: zero rows, one row, and zip comments up to the largest length");

    const char *path = "out/test_npz_columns_adversarial.npz";
    DataFrame no_rows = df_new(0);
    Vec none = mat_new(0, 1);
    df_add_numeric_col(&no_rows, "x", none);
    df_add_string_col(&no_rows, "s", NULL);
    df_write_npz(&no_rows, path);
    const char *only_s[1] = { "s" };
    DataFrame no_rows_back = df_read_npz_columns(path, only_s, 1);
    assert(no_rows_back.r == 0 && no_rows_back.n_cols == 1 && no_rows_back.columns[0].type == COL_STRING);
    df_free(&no_rows_back);

    DataFrame single = df_new(1);
    Vec one = mat_lit(1, 1, 99.f);
    df_add_numeric_col(&single, "x", one);
    df_add_numeric_col(&single, "y", one);
    df_write_npz_compressed(&single, path);
    const char *only_y[1] = { "y" };
    DataFrame single_back = df_read_npz_columns(path, only_y, 1);
    assert(single_back.r == 1 && single_back.n_cols == 1 && AT(single_back.numeric, 0, 0) == 99.f);
    df_free(&single_back);

    /* A trailing comment moves the end-of-directory record away from the
       end of the file. At 65535 bytes, the most the 16-bit length allows,
       the record is the first byte of the tail the reader takes from the
       file; past it the record is out of reach by the format's own rule. */
    long valid_len;
    df_write_npz(&single, path);
    unsigned char *valid = (unsigned char*)frame_read_file(path, &valid_len);
    unsigned comment_lengths[4] = { 1, 1000, 65534, 65535 };
    for (int t = 0; t < 4; t++) {
        size_t total = (size_t)valid_len + comment_lengths[t];
        unsigned char *with_comment = (unsigned char*)malloc(total);
        memcpy(with_comment, valid, (size_t)valid_len);
        memset(with_comment + valid_len, 'A', comment_lengths[t]);
        put16(with_comment + valid_len - 2, comment_lengths[t]);
        write_bytes(path, with_comment, total);
        DataFrame full = df_read_npz(path);
        assert(df_identical(&full, &single));
        DataFrame back = df_read_npz_columns(path, only_y, 1);
        assert(back.n_cols == 1 && AT(back.numeric, 0, 0) == 99.f);
        df_free(&full);
        df_free(&back);
        free(with_comment);
    }
    free(valid);

    mat_free(none); mat_free(one);
    df_free(&no_rows); df_free(&single);
    remove(path);
}

static const char *g_columns_path = "out/test_npz_columns_bad.npz";
static const char *g_columns_x[1] = { "x" };
static const char *g_columns_y[1] = { "y" };
static void call_read_y(void) { DataFrame df = df_read_npz_columns(g_columns_path, g_columns_y, 1); (void)df; }
static void call_read_full(void) { DataFrame df = df_read_npz(g_columns_path); (void)df; }
static void call_read_none(void) { DataFrame df = df_read_npz_columns(g_columns_path, g_columns_x, 0); (void)df; }
static void call_read_twice(void) {
    const char *names[3] = { "x", "y", "x" };
    DataFrame df = df_read_npz_columns(g_columns_path, names, 3);
    (void)df;
}
static void call_read_row_names_key(void) {
    const char *names[2] = { "x", FRAME_NPZ_ROW_NAMES_KEY };
    DataFrame df = df_read_npz_columns(g_columns_path, names, 2);
    (void)df;
}
static void call_read_short_and_long(void) {
    const char *names[2] = { "short", "long" };
    DataFrame df = df_read_npz_columns(g_columns_path, names, 2);
    (void)df;
}

/* A member that is not asked for is never read: a corrupted one, one of
   another length and one of a dtype this build cannot read all go
   unnoticed, while asking for it, or reading the whole archive, still
   fails. This is what separates a reader that skips members from one that
   reads them all and drops some afterwards. */
static void test_column_subset_reads_only_what_it_asks_for(void) {
    puts("df_read_npz_columns: an unasked member is never read, an asked one is checked");

    DataFrame df = df_new(40);
    Vec x = mat_new(40, 1), y = mat_new(40, 1);
    for (int i = 0; i < 40; i++) { x.d[i] = (mreal)i; y.d[i] = (mreal)(100 + i); }
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "y", y);
    df_write_npz(&df, g_columns_path);

    /* flip a value byte of y, the second member */
    long len;
    unsigned char *bytes = (unsigned char*)frame_read_file(g_columns_path, &len);
    uint32_t directory = frame_npz_u32(bytes + len - 22 + 16);
    unsigned first_name = frame_npz_u16(bytes + directory + 28);
    uint32_t y_local = frame_npz_u32(bytes + directory + 46 + first_name + 42);
    size_t y_data = y_local + 30 + frame_npz_u16(bytes + y_local + 26) + frame_npz_u16(bytes + y_local + 28);
    bytes[y_data + 128 + 4] ^= 0xFF;
    write_bytes(g_columns_path, bytes, (size_t)len);
    free(bytes);

    DataFrame just_x = df_read_npz_columns(g_columns_path, g_columns_x, 1);
    assert(just_x.n_cols == 1 && just_x.r == 40);
    for (int i = 0; i < 40; i++) assert(AT(just_x.numeric, i, 0) == (mreal)i);
    df_free(&just_x);
    expect_abort(call_read_y);
    expect_abort(call_read_full);

    /* members disagreeing about the row count, only one of them asked for */
    mreal two[2] = { 1, 2 }, three[3] = { 1, 2, 3 };
    FixtureArray lengths[2] = {
        { "short", frame_npy_mreal_descr(), two, sizeof two, 2, 1 },
        { "long", frame_npy_mreal_descr(), three, sizeof three, 3, 1 }
    };
    build_fixture_npz(g_columns_path, lengths, 2);
    const char *only_short[1] = { "short" };
    DataFrame short_back = df_read_npz_columns(g_columns_path, only_short, 1);
    assert(short_back.r == 2 && short_back.n_cols == 1 && AT(short_back.numeric, 1, 0) == 2);
    df_free(&short_back);
    expect_abort(call_read_short_and_long);

    /* a '<i4' member beside a readable one */
    int32_t ints[2] = { 1, 2 };
    FixtureArray dtypes[2] = {
        { "y", "<i4", ints, sizeof ints, 2, 1 },
        { "x", frame_npy_mreal_descr(), two, sizeof two, 2, 1 }
    };
    build_fixture_npz(g_columns_path, dtypes, 2);
    DataFrame x_back = df_read_npz_columns(g_columns_path, g_columns_x, 1);
    assert(x_back.r == 2 && x_back.n_cols == 1);
    df_free(&x_back);
    expect_abort(call_read_y);

    mat_free(x); mat_free(y);
    df_free(&df);
    remove(g_columns_path);
}

static void test_column_subset_contract_aborts(void) {
    puts("df_read_npz_columns contract violations abort (no names, a name twice, the row-label key)");
    DataFrame df = df_new(2);
    Vec x = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "y", x);
    df_write_npz(&df, g_columns_path);
    expect_abort(call_read_none);
    expect_abort(call_read_twice);
    expect_abort(call_read_row_names_key);
    /* the same archive is readable, so the aborts above are the contract */
    DataFrame back = df_read_npz_columns(g_columns_path, g_columns_x, 1);
    assert(back.n_cols == 1);
    df_free(&back);
    mat_free(x);
    df_free(&df);
    remove(g_columns_path);
}

/* --- STRESS=1: fixed-seed randomized round-trip fuzzing over frames
   with numeric columns, string columns and row labels together, the same
   technique test_npy.c/test_csv.c/test_json.c use. Comparison is exact
   for the numeric side (raw bytes, no text formatting anywhere in the
   path) and strcmp for the string side. --- */

static const char *random_string_pool[] = {
    "", "a", "north", "1973 March", "caff\xc3\xa8", "\xe4\xb8\xad\xe6\x96\x87",
    "\xf0\x9f\x98\x80", "a longer label with spaces", "-12.5", "NA"
};

static DataFrame random_dataframe(void) {
    int r = 1 + rand() % 8;
    int c = 1 + rand() % 5;
    DataFrame df = df_new(r);
    for (int j = 0; j < c; j++) {
        char name[16];
        snprintf(name, sizeof name, "c%d", j);
        if (rand() % 2) {
            Vec v = mat_new(r, 1);
            for (int i = 0; i < r; i++) {
                switch (rand() % 5) {
                    case 0: v.d[i] = 0; break;
                    case 1: v.d[i] = -((mreal)(rand() % 100000)) / 7; break;
                    case 2: v.d[i] = ((mreal)(rand() % 100000)) / 3; break;
                    case 3: v.d[i] = (mreal)(rand() % 2 ? 1 : -1) * (mreal)1e30; break;
                    default: v.d[i] = (mreal)1e-30; break;
                }
            }
            df_add_numeric_col(&df, name, v);
            mat_free(v);
        } else {
            const char **col = (const char**)malloc((size_t)r * sizeof(char*));
            int pool = (int)(sizeof random_string_pool / sizeof *random_string_pool);
            for (int i = 0; i < r; i++) col[i] = random_string_pool[rand() % pool];
            df_add_string_col(&df, name, col);
            free(col);
        }
    }
    if (rand() % 2) {
        const char **names = (const char**)malloc((size_t)r * sizeof(char*));
        int pool = (int)(sizeof random_string_pool / sizeof *random_string_pool);
        for (int i = 0; i < r; i++) names[i] = random_string_pool[rand() % pool];
        df_set_row_names(&df, names);
        free(names);
    }
    return df;
}

static int df_identical(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->n_cols != b->n_cols || a->n_string != b->n_string) return 0;
    if ((a->row_names == NULL) != (b->row_names == NULL)) return 0;
    for (int j = 0; j < a->n_cols; j++) {
        if (a->columns[j].type != b->columns[j].type) return 0;
        if (strcmp(a->columns[j].name, b->columns[j].name) != 0) return 0;
        if (a->columns[j].type == COL_NUMERIC) {
            for (int i = 0; i < a->r; i++)
                if (AT(a->numeric, i, a->columns[j].index) != AT(b->numeric, i, b->columns[j].index)) return 0;
        } else {
            for (int i = 0; i < a->r; i++)
                if (strcmp(a->string_cols[a->columns[j].index][i], b->string_cols[b->columns[j].index][i]) != 0) return 0;
        }
    }
    if (a->row_names)
        for (int i = 0; i < a->r; i++)
            if (strcmp(a->row_names[i], b->row_names[i]) != 0) return 0;
    return 1;
}

static void test_random_roundtrip_stress(void) {
    puts("  random mixed-column round-trip fuzzing (fixed seed, unicode and fragile magnitudes)");
    srand(46);
    const char *path = "out/test_npz_fuzz.npz";
    for (int trial = 0; trial < 200; trial++) {
        DataFrame original = random_dataframe();
        /* both writers over the same corpus: the deflated path has to
           mean exactly what the stored one does at every shape */
        if (trial % 2) df_write_npz_compressed(&original, path);
        else df_write_npz(&original, path);
        DataFrame back = df_read_npz(path);
        assert(df_identical(&original, &back));
        df_free(&original);
        df_free(&back);
    }
    remove(path);
    printf("  200 random mixed DataFrames round-tripped ok (stored and deflated alternating)\n");
}

/* Random frames, each read back through random subsets of its columns in
   random order, some with a name the archive does not hold, against the
   full read of the same file. */
static void test_random_column_subset_stress(void) {
    puts("  random column subsets against the full read (fixed seed)");
    srand(48);
    const char *path = "out/test_npz_columns_fuzz.npz";
    const char *absent = "not_a_column";
    for (int trial = 0; trial < 200; trial++) {
        DataFrame original = random_dataframe();
        if (trial % 2) df_write_npz_compressed(&original, path);
        else df_write_npz(&original, path);
        DataFrame full = df_read_npz(path);
        for (int draw = 0; draw < 5; draw++) {
            const char *names[8];
            int n_names = 0;
            int order[8];
            for (int k = 0; k < original.n_cols; k++) order[k] = k;
            for (int k = original.n_cols - 1; k > 0; k--) {
                int l = rand() % (k + 1);
                int t = order[k]; order[k] = order[l]; order[l] = t;
            }
            int take = 1 + rand() % original.n_cols;
            for (int k = 0; k < take; k++) names[n_names++] = original.columns[order[k]].name;
            if (rand() % 3 == 0) names[n_names++] = absent;
            DataFrame some = df_read_npz_columns(path, names, n_names);
            assert(matches_full_read(&some, &full, names, n_names));
            df_free(&some);
        }
        df_free(&original);
        df_free(&full);
    }
    remove(path);
    printf("  200 random DataFrames, 5 subsets each, agreed with the full read\n");
}

int main(void) {
    frame_mkdir_p("out");
    test_numeric_roundtrip();
    test_mixed_columns_preserve_declaration_order();
    test_row_names_roundtrip();
    test_unicode_strings_roundtrip();
    test_reads_real_numpy_archives();
    test_adversarial_shapes();
    test_compressed_archive_means_the_same_thing();
    test_rebuilt_metadata_drives_sql_and_join();
    test_malformed_npz_aborts();
    test_write_contract_aborts();
    test_column_subset_known_values();
    test_column_subset_agrees_with_full_read();
    test_column_subset_reads_real_numpy_archives();
    test_column_subset_adversarial_shapes();
    test_column_subset_reads_only_what_it_asks_for();
    test_column_subset_contract_aborts();

    if (getenv("STRESS")) {
        test_random_roundtrip_stress();
        test_random_column_subset_stress();
    }

    puts("test_npz: all passed");
    return 0;
}
