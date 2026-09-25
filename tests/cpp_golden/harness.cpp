// Golden-output generator for porting Breeze-TTS-2.cpp's text_split.cpp + ws_api.cpp's
// drain logic to Python. Links against the real text_split.cpp for split_text/split_sentences/
// split_clauses, and embeds a byte-for-byte copy of sentence_end/drain from ws_api.cpp
// (apps/server/ws_api.cpp, lines ~60-95) so we exercise the exact same buffer-drain code the
// server runs, without pulling in the rest of the server's networking machinery.
#include "breeze/generation.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace breeze {

// ---- verbatim copy of apps/server/ws_api.cpp's sentence_end + drain (lines ~60-95) ----

static bool sentence_end(const std::string & s, size_t i) {
    const char c = s[i];
    if (c == '.' || c == '!' || c == '?' || c == ';') {
        // a bare dot inside a number or an abbreviation is not the end of anything
        return i + 1 >= s.size() || s[i + 1] == ' ' || s[i + 1] == '\n';
    }
    // the cjk stops carry their own spacing
    static const char * stops[] = { "\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f", "\xef\xbc\x9b" };
    for (const char * st : stops)
        if (s.compare(i, 3, st) == 0) return true;
    return false;
}

// moves whole sentences out of buf, leaving a trailing partial behind unless force is set
static std::vector<std::string> drain(std::string & buf, int budget, bool force) {
    std::vector<std::string> out;
    size_t cut = 0;
    for (size_t i = 0; i < buf.size(); i++)
        if (sentence_end(buf, i)) cut = i + ((unsigned char) buf[i] < 0x80 ? 1 : 3);

    // nothing finished but the buffer is already long enough to speak, so break it on a space
    if (!cut && !force && (int) buf.size() > budget) {
        const size_t sp = buf.rfind(' ');
        if (sp != std::string::npos) cut = sp + 1;
    }
    if (force) cut = buf.size();
    if (!cut) return out;

    std::string ready = buf.substr(0, cut);
    buf.erase(0, cut);
    for (std::string & p : split_text(ready, budget)) {
        while (!p.empty() && p.front() == ' ') p.erase(p.begin());
        if (!p.empty()) out.push_back(p);
    }
    return out;
}

// ---- JSON helpers (small hand-rolled escaper, mirrors ws_api.cpp's esc()) ----

static std::string json_esc(const std::string & s) {
    std::string o;
    char esc[8];
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char) c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') o += "\\r";
        // Any other control byte (notably a literal NUL, used by the nul_byte_in_text
        // golden below) is invalid unescaped inside a JSON string, so \u-escape it
        // rather than emitting the raw byte.
        else if (c < 0x20) { std::snprintf(esc, sizeof(esc), "\\u%04x", c); o += esc; }
        else o += (char) c;
    }
    return o;
}

static void print_str_array(const std::vector<std::string> & v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) std::cout << ",";
        std::cout << "\"" << json_esc(v[i]) << "\"";
    }
    std::cout << "]";
}

struct SplitCase {
    const char * name;
    std::string text;
    int budget;
    int first_budget; // 0 means "unset" (matches the C++ default)
};

struct DrainCase {
    const char * name;
    std::string buffer;
    int budget;
    bool force;
};

int main() {
    // A literal NUL byte embedded in the text. `"a.\x00 b c..."` alone would truncate at
    // std::string's implicit strlen-based construction, so build it from an explicit length
    // (sizeof of the array literal includes the embedded NUL, minus the compiler-added
    // terminator) to keep the NUL as real payload instead of an end-of-string marker.
    static const char nul_case_bytes[] = "a.\x00 b c d e f g h i j k l m n";
    const std::string nul_case_text(nul_case_bytes, sizeof(nul_case_bytes) - 1);

    std::vector<SplitCase> split_cases = {
        {"english_multi_sentence",
         "The quick brown fox jumps over the lazy dog. It was a sunny day! Are you coming? Yes.",
         30, 0},
        {"sentence_longer_than_budget_clause_split",
         "This is a single very long sentence with no terminal punctuation yet, containing several commas, which should force clause splitting, because it exceeds the budget by quite a lot in total weight",
         40, 0},
        {"long_run_no_punctuation_space_split",
         "supercalifragilisticexpialidocious word another word yetanotherword andmore words continuing on and on without any commas or periods anywhere in this run at all",
         30, 0},
        {"chinese_period_comma",
         "\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa4\xa9\xe6\xb0\x94\xe5\xbe\x88\xe5\xa5\xbd\xef\xbc\x8c\xe6\x88\x91\xe4\xbb\xac\xe5\x8e\xbb\xe5\x85\xac\xe5\x9b\xad\xe6\x95\xa3\xe6\xad\xa5\xe3\x80\x82"
         "\xe4\xb8\x8b\xe5\x8d\x88\xe5\x8f\xaf\xe8\x83\xbd\xe4\xbc\x9a\xe4\xb8\x8b\xe9\x9b\xa8\xef\xbc\x8c\xe6\x89\x80\xe4\xbb\xa5\xe6\x88\x91\xe4\xbb\xac\xe5\xbe\x97\xe5\xb8\xa6\xe4\xbc\x9e\xe3\x80\x82",
         20, 0},
        {"mixed_english_chinese",
         "Hello world, \xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c, this is a mixed sentence with \xe4\xb8\xad\xe6\x96\x87 characters thrown in\xe3\x80\x82",
         25, 0},
        {"closing_quote_after_period",
         "She said \"this is great.\" Then she left. He replied (with a nod.) and walked off.",
         30, 0},
        {"ellipsis",
         "Wait for it\xe2\x80\xa6 here it comes\xe2\x80\xa6 finally!",
         15, 0},
        {"newlines",
         "First line\nSecond line\nThird line is a bit longer than the others\nFourth",
         25, 0},
        {"first_budget_smaller_than_budget",
         "The quick brown fox jumps over the lazy dog. It was a sunny day! Are you coming? Yes indeed.",
         30, 10},
        {"budget_zero_no_split",
         "This text should not be split at all regardless of its length because budget is zero.",
         0, 0},
        {"empty_string", "", 20, 0},
        {"whitespace_only", "   \n   ", 20, 0},
        // Short sentences (unlike first_budget_smaller_than_budget's one long sentence that
        // never actually gets clause-split) so first_budget caps the opening piece and later
        // pieces genuinely reset to the wider budget.
        {"budget_reset_smaller_first_budget",
         "The quick brown. fox jumps. over the lazy dog. It was a sunny day! Are you coming?",
         30, 10},
        {"budget_reset_larger_first_budget",
         "The quick brown. fox jumps. over the lazy dog. It was a sunny day! Are you coming?",
         20, 60},
        // 4-byte (emoji) and 2-byte (accented Latin) multibyte chars mixed with ASCII.
        {"emoji_and_accented_multibyte",
         "\xf0\x9f\x98\x80\xf0\x9f\x98\x80\xf0\x9f\x98\x80 ab, \xf0\x9f\x98\x80\xf0\x9f\x98\x80 cd, "
         "\xc3\xa9\xc3\xa9 ff, \xf0\x9f\x98\x80\xf0\x9f\x98\x80\xf0\x9f\x98\x80\xf0\x9f\x98\x80 gg.",
         10, 0},
        {"nul_byte_in_text", nul_case_text, 5, 0},
    };

    std::vector<DrainCase> drain_cases = {
        {"drain_finished_sentence_not_forced",
         "This is done. This is not",
         30, false},
        {"drain_no_sentence_end_over_budget_space_split",
         "this buffer has no terminal punctuation at all but it is definitely longer than the budget so",
         30, false},
        {"drain_forced_takes_everything",
         "partial sentence with no end",
         30, true},
        {"drain_nothing_ready",
         "short partial",
         30, false},
        {"drain_leading_spaces_trimmed",
         "Done first.   Second one is also done.  trailing partial",
         30, false},
        {"drain_chinese_sentence_end",
         "\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa4\xa9\xe6\xb0\x94\xe5\xbe\x88\xe5\xa5\xbd\xe3\x80\x82\xe6\xb2\xa1\xe5\x86\x99\xe5\xae\x8c\xe7\x9a\x84\xe9\x83\xa8\xe5\x88\x86",
         20, false},
        {"drain_empty_buffer_forced",
         "", 30, true},
        {"drain_ellipsis_not_recognized_by_sentence_end",
         "Wait for it\xe2\x80\xa6 more words keep coming without any real stop here at all so it runs long",
         30, false},
        // 2-byte-per-char cyrillic text: 30 chars * weight 3 = 90 (over budget by weigh()) but
        // only 56 UTF-8 bytes (under budget 60) -- drain's fallback compares raw buf.size(), not
        // weigh(), so this stays undrained even though split_text would call it over-budget.
        {"drain_byte_vs_weight_asymmetry_undrained",
         "\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xba\xd0\xb0\xd0\xba \xd0\xb4\xd0\xb5\xd0\xbb\xd0\xb0 \xd1\x81\xd0\xb5\xd0\xb3\xd0\xbe\xd0\xb4\xd0\xbd\xd1\x8f \xd1\x85\xd0\xbe\xd1\x80\xd0\xbe\xd1\x88\xd0\xbe",
         60, false},
        // same buffer, smaller budget so the raw byte length (56) does exceed it and the
        // space-fallback cut fires.
        {"drain_byte_vs_weight_asymmetry_drained",
         "\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xba\xd0\xb0\xd0\xba \xd0\xb4\xd0\xb5\xd0\xbb\xd0\xb0 \xd1\x81\xd0\xb5\xd0\xb3\xd0\xbe\xd0\xb4\xd0\xbd\xd1\x8f \xd1\x85\xd0\xbe\xd1\x80\xd0\xbe\xd1\x88\xd0\xbe",
         40, false},
        // ws_api.cpp's sentence_end doesn't skip closing quotes/brackets after '.!?;' (unlike
        // split_sentences), so the quote right after "done." blocks the break: nothing drains.
        {"drain_no_quote_absorption",
         "He said \"done.\" next",
         30, false},
        // ws_api.cpp's stops[] (see sentence_end above) omits the fullwidth period, so it
        // isn't a sentence end here even though split_sentences' _CJK_SENTENCE_STOPS has it.
        {"drain_fullwidth_period_not_sentence_end",
         "Done\xef\xbc\x8enext",
         30, false},
    };

    // Each case's inputs are emitted alongside its result so gen_goldens.py has a single
    // source of truth (golden.json) instead of a second, hand-kept input table that could
    // silently drift out of sync with this one.
    std::cout << "{\"split_text\":[";
    for (size_t i = 0; i < split_cases.size(); i++) {
        auto & c = split_cases[i];
        if (i) std::cout << ",";
        auto result = split_text(c.text, c.budget, c.first_budget);
        std::cout << "{\"name\":\"" << c.name << "\""
                   << ",\"text\":\"" << json_esc(c.text) << "\""
                   << ",\"budget\":" << c.budget
                   << ",\"first_budget\":" << c.first_budget
                   << ",\"result\":";
        print_str_array(result);
        std::cout << "}";
    }
    std::cout << "],\"drain\":[";
    for (size_t i = 0; i < drain_cases.size(); i++) {
        auto & c = drain_cases[i];
        if (i) std::cout << ",";
        std::string buf = c.buffer;
        auto pieces = drain(buf, c.budget, c.force);
        std::cout << "{\"name\":\"" << c.name << "\""
                   << ",\"buffer\":\"" << json_esc(c.buffer) << "\""
                   << ",\"budget\":" << c.budget
                   << ",\"force\":" << (c.force ? "true" : "false")
                   << ",\"pieces\":";
        print_str_array(pieces);
        std::cout << ",\"remaining\":\"" << json_esc(buf) << "\"}";
    }
    std::cout << "]}\n";
    return 0;
}

} // namespace breeze

int main() { return breeze::main(); }
