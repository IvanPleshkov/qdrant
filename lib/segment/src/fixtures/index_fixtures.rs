extern crate profiler_proc_macro;
use profiler_proc_macro::trace;

use crate::payload_storage::ConditionChecker;
use crate::spaces::metric::Metric;
use crate::types::{Filter, PointOffsetType, VectorElementType};
use crate::vector_storage::simple_vector_storage::SimpleRawScorer;
use bit_vec::BitVec;
use itertools::Itertools;
use rand::Rng;

#[trace]
pub fn random_vector<R: Rng + ?Sized>(rnd_gen: &mut R, size: usize) -> Vec<VectorElementType> {
    (0..size).map(|_| rnd_gen.gen_range(0.0..1.0)).collect()
}

pub struct FakeConditionChecker {}

impl ConditionChecker for FakeConditionChecker {
    fn check(&self, _point_id: PointOffsetType, _query: &Filter) -> bool {
        true
    }
}

pub struct TestRawScorerProducer<TMetric: Metric> {
    pub dim: usize,
    pub vectors: Vec<VectorElementType>,
    pub deleted: BitVec,
    pub metric: TMetric,
}

impl<TMetric> TestRawScorerProducer<TMetric>
where
    TMetric: Metric,
{
    #[trace]
    pub fn new<R>(dim: usize, num_vectors: usize, metric: TMetric, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let mut vectors = Vec::new();
        for _ in 0..num_vectors {
            let rnd_vec = random_vector(rng, dim);
            let rnd_vec = metric.preprocess(&rnd_vec).unwrap_or(rnd_vec);
            vectors.extend_from_slice(rnd_vec.as_slice());
        }

        TestRawScorerProducer {
            dim,
            vectors,
            deleted: BitVec::from_elem(num_vectors, false),
            metric,
        }
    }

    #[trace]
    pub fn get_raw_scorer(&self, query: Vec<VectorElementType>) -> SimpleRawScorer<TMetric> {
        SimpleRawScorer {
            dim: self.dim,
            query: self.metric.preprocess(&query).unwrap_or(query),
            metric: &self.metric,
            vectors: &self.vectors,
            deleted: &self.deleted,
        }
    }
}
