//! Categorical type system implementation

pub mod objects;
pub mod morphisms;
pub mod composition;
pub mod builder;
pub mod zero_object;

pub use objects::{Empty, Unit, Numeric};
pub use morphisms::{Morphism, Genesis, Instantiation, Identity, NumericMorphism};
pub use composition::{Compose, Composition, CompositionResult};
pub use builder::{MorphismBuilder, GenesisBuilder};
pub use zero_object::{
    ZeroObject, EmptyAspect, InfiniteAspect, ModelParameters,
    eta, zeta, sample_from_empty, update_infinite,
};