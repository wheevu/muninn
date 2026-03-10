#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub end_offset: usize,
}

impl Span {
    pub const fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
            end_line: line,
            end_column: column,
            end_offset: offset,
        }
    }

    pub const fn range(
        line: usize,
        column: usize,
        offset: usize,
        end_line: usize,
        end_column: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            line,
            column,
            offset,
            end_line,
            end_column,
            end_offset,
        }
    }

    pub const fn width(&self) -> usize {
        self.end_offset.saturating_sub(self.offset)
    }
}
