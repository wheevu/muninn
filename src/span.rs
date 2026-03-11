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

    pub fn contains_offset(&self, offset: usize) -> bool {
        self.offset <= offset && offset < self.end_offset.max(self.offset + 1)
    }

    pub fn merge(self, other: Span) -> Span {
        if self.line == 0 {
            return other;
        }
        if other.line == 0 {
            return self;
        }
        Span {
            line: self.line,
            column: self.column,
            offset: self.offset,
            end_line: other.end_line,
            end_column: other.end_column,
            end_offset: other.end_offset,
        }
    }
}
