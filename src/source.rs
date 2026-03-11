pub fn compute_line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (offset, ch) in source.char_indices() {
        if ch == '\n' {
            starts.push(offset + ch.len_utf8());
        }
    }
    starts
}

pub fn offset_to_utf16_position(source: &str, line_starts: &[usize], offset: usize) -> (u32, u32) {
    let clamped = offset.min(source.len());
    let line = line_for_offset(line_starts, clamped);
    let line_start = line_starts.get(line).copied().unwrap_or(0);
    let character = source[line_start..clamped]
        .chars()
        .map(|ch| ch.len_utf16() as u32)
        .sum();
    (line as u32, character)
}

pub fn utf16_position_to_offset(
    source: &str,
    line_starts: &[usize],
    line: u32,
    character: u32,
) -> Option<usize> {
    let line = line as usize;
    let start = *line_starts.get(line)?;
    let end = line_starts
        .get(line + 1)
        .copied()
        .unwrap_or(source.len());
    let mut current = start;
    let mut utf16_units = 0u32;

    for ch in source[start..end].chars() {
        if utf16_units >= character {
            return Some(current);
        }
        utf16_units += ch.len_utf16() as u32;
        current += ch.len_utf8();
        if utf16_units == character {
            return Some(current);
        }
    }

    if utf16_units <= character {
        Some(end)
    } else {
        None
    }
}

pub fn offset_to_line_column(source: &str, line_starts: &[usize], offset: usize) -> (usize, usize) {
    let clamped = offset.min(source.len());
    let line = line_for_offset(line_starts, clamped);
    let line_start = line_starts.get(line).copied().unwrap_or(0);
    let column = source[line_start..clamped].chars().count() + 1;
    (line + 1, column)
}

fn line_for_offset(line_starts: &[usize], offset: usize) -> usize {
    line_starts.partition_point(|start| *start <= offset).saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::{compute_line_starts, offset_to_utf16_position, utf16_position_to_offset};

    #[test]
    fn round_trips_utf16_positions() {
        let source = "let bird = \"🐦\";\nprint(bird);\n";
        let lines = compute_line_starts(source);
        let offset = source.find("🐦").expect("bird offset");
        let (line, character) = offset_to_utf16_position(source, &lines, offset);
        assert_eq!((line, character), (0, 12));
        let round_trip = utf16_position_to_offset(source, &lines, line, character).expect("offset");
        assert_eq!(round_trip, offset);
    }
}
