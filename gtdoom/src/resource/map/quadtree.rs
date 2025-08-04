#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadTreeDimensions {
    chunk_size: u32,
    max_size: u32,
}

impl QuadTreeDimensions {
    pub fn new(chunk_size: u32, min_size: u32) -> Self {
        assert!(min_size < i32::MAX as u32);
        assert!((chunk_size * 2) < i32::MAX as u32);
        let mut max_size = chunk_size * 2;
        while max_size < min_size {
            max_size *= 2;
        }
        Self {
            chunk_size,
            max_size,
        }
    }

    #[inline]
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    #[inline]
    pub fn bounds(&self) -> Bounds {
        Bounds {
            min: glm::vec2(-(self.max_size as i32), -(self.max_size as i32)),
            max: glm::vec2(self.max_size as i32, self.max_size as i32),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bounds {
    pub min: glm::TVec2<i32>,
    pub max: glm::TVec2<i32>,
}

impl Bounds {
    fn is_leaf(&self, chunk_size: u32) -> bool {
        (self.max.x - self.min.x).unsigned_abs() < chunk_size
            && (self.max.y - self.min.y).unsigned_abs() < chunk_size
    }

    fn subdivide(self, pos: glm::TVec2<i32>) -> Option<(usize, Bounds)> {
        let center = glm::vec2(
            self.min.x + ((self.max.x - self.min.x) / 2),
            self.min.y + ((self.max.y - self.min.y) / 2),
        );
        if pos.x < self.min.x || pos.x >= self.max.x || pos.y < self.min.y || pos.y >= self.max.y {
            None
        } else if pos.x < center.x {
            if pos.y < center.y {
                Some((0, Bounds {
                    min: self.min,
                    max: center,
                }))
            } else {
                Some((1, Bounds {
                    min: glm::vec2(self.min.x, center.y),
                    max: glm::vec2(center.x, self.max.y),
                }))
            }
        } else {
            if pos.y < center.y {
                Some((2, Bounds {
                    min: glm::vec2(center.x, self.min.y),
                    max: glm::vec2(self.max.x, center.y),
                }))
            } else {
                Some((3, Bounds {
                    min: center,
                    max: self.max,
                }))
            }
        }
    }
}

#[derive(Debug)]
enum Node<T> {
    Empty,
    Branch(Box<[Node<T>; 4]>),
    Leaf(Box<T>),
}

impl<T> Default for Node<T> {
    #[inline]
    fn default() -> Self {
        Self::Empty
    }
}

impl<T> Node<T> {
    fn branch() -> Self {
        Self::Branch(Box::new(std::array::from_fn(|_| Node::default())))
    }
}

pub struct QuadTree<T> {
    dims: QuadTreeDimensions,
    inner: Node<T>,
}

impl<T> QuadTree<T> {
    pub fn new(dims: QuadTreeDimensions) -> Self {
        Self {
            dims,
            inner: Default::default(),
        }
    }

    pub fn get(&self, pos: glm::TVec2<i32>) -> Option<&T> {
        let mut node = &self.inner;
        let mut sub_node_idx;
        let mut bounds = self.dims.bounds();

        loop {
            match node {
                Node::Empty => return None,
                Node::Branch(sub_nodes) => {
                    (sub_node_idx, bounds) = bounds.subdivide(pos)?;
                    node = &sub_nodes[sub_node_idx];
                },
                Node::Leaf(value) => return Some(value.as_ref()),
            }
        }
    }

    pub fn get_mut(&mut self, pos: glm::TVec2<i32>) -> Option<&mut T> {
        let mut node = &mut self.inner;
        let mut sub_node_idx;
        let mut bounds = self.dims.bounds();

        loop {
            match node {
                Node::Empty => return None,
                Node::Branch(sub_nodes) => {
                    (sub_node_idx, bounds) = bounds.subdivide(pos)?;
                    node = &mut sub_nodes[sub_node_idx];
                },
                Node::Leaf(value) => return Some(value.as_mut()),
            }
        }
    }

    pub fn get_or_insert_with<F>(&mut self, pos: glm::TVec2<i32>, f: F) -> Option<&mut T>
    where
        F: FnOnce() -> T,
    {
        let mut node = &mut self.inner;
        let mut sub_node_idx;
        let mut bounds = self.dims.bounds();

        loop {
            match node {
                Node::Empty => {
                    if bounds.is_leaf(self.dims.chunk_size) {
                        *node = Node::Leaf(Box::new(f()));
                        if let Node::Leaf(value) = node {
                            return Some(value.as_mut())
                        } else {
                            unreachable!()
                        }
                    } else {
                        *node = Node::branch();
                        (sub_node_idx, bounds) = bounds.subdivide(pos)?;
                        if let Node::Branch(sub_nodes) = node {
                            node = &mut sub_nodes[sub_node_idx];
                        } else {
                            unreachable!()
                        }
                    }
                },
                Node::Branch(sub_nodes) => {
                    (sub_node_idx, bounds) = bounds.subdivide(pos)?;
                    node = &mut sub_nodes[sub_node_idx];
                },
                Node::Leaf(value) => return Some(value.as_mut()),
            }
        }
    }

    #[inline]
    pub fn get_or_insert(&mut self, pos: glm::TVec2<i32>, value: T) -> Option<&mut T> {
        self.get_or_insert_with(pos, || value)
    }
}

impl<T: Default> QuadTree<T> {
    #[inline]
    pub fn get_or_default(&mut self, pos: glm::TVec2<i32>) -> Option<&mut T> {
        self.get_or_insert_with(pos, T::default)
    }
}