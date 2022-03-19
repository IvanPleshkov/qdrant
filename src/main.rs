extern crate profiler_proc_macro;
use profiler_proc_macro::trace;

#[cfg(feature = "web")]
mod actix;
pub mod common;
#[cfg(feature = "consensus")]
mod consensus;
mod settings;
mod tonic;

#[cfg(feature = "consensus")]
use consensus::Consensus;
#[cfg(feature = "consensus")]
use slog::Drain;

/*
use tracy_client::*;
#[global_allocator]
static GLOBAL: ProfiledAllocator<std::alloc::System> =
    ProfiledAllocator::new(std::alloc::System, 100);
*/

use rand::{thread_rng, Rng};
use segment::fixtures::index_fixtures::{
    random_vector, FakeConditionChecker, TestRawScorerProducer,
};
use segment::index::hnsw_index::graph_layers::GraphLayers;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::spaces::simple::CosineMetric;
use segment::types::PointOffsetType;

use std::time::{Duration, Instant};

const NUM_VECTORS: usize = 250;
const DIM: usize = 32;
const M: usize = 16;
const TOP: usize = 10;
const EF_CONSTRUCT: usize = 64;
const EF: usize = 64;
const USE_HEURISTIC: bool = true;

#[trace]
fn build_index(num_vectors: usize) -> (TestRawScorerProducer<CosineMetric>, GraphLayers) {
    let mut rng = thread_rng();

    let vector_holder = TestRawScorerProducer::new(DIM, num_vectors, CosineMetric {}, &mut rng);
    let mut graph_layers = GraphLayers::new(num_vectors, M, M * 2, EF_CONSTRUCT, 10, USE_HEURISTIC);
    let fake_condition_checker = FakeConditionChecker {};
    for idx in 0..(num_vectors as PointOffsetType) {
        let added_vector = vector_holder.vectors[idx as usize].to_vec();
        let raw_scorer = vector_holder.get_raw_scorer(added_vector);
        let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
        let level = graph_layers.get_random_layer(&mut rng);
        graph_layers.link_new_point(idx, level, &scorer);
    }
    (vector_holder, graph_layers)
}

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let mut rng = thread_rng();

    let (vector_holder, graph_layers) = build_index(NUM_VECTORS);

    {
        let fake_condition_checker = FakeConditionChecker {};
        let query = random_vector(&mut rng, DIM);
        let raw_scorer = vector_holder.get_raw_scorer(query);
        let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
        graph_layers.search(TOP, EF, &scorer);
    }

    for _ in 0..10 {
        let fake_condition_checker = FakeConditionChecker {};
        let query = random_vector(&mut rng, DIM);
        let raw_scorer = vector_holder.get_raw_scorer(query);
        let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
        graph_layers.search(TOP, EF, &scorer);
    }

    /*
        let (vector_holder, graph_layers) = build_index(NUM_VECTORS * 10);

        {
            let fake_condition_checker = FakeConditionChecker {};
            let query = random_vector(&mut rng, DIM);
            let raw_scorer = vector_holder.get_raw_scorer(query);
            let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
            graph_layers.search(TOP, EF, &scorer);
        }

        {
            let fake_condition_checker = FakeConditionChecker {};
            let query = random_vector(&mut rng, DIM);
            let raw_scorer = vector_holder.get_raw_scorer(query);
            let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);

            let mut points_to_score = (0..1500).map(|_| rng.gen_range(0..(NUM_VECTORS * 10)) as u32);
            scorer.score_iterable_points(&mut points_to_score, 1000, |_| {})
        }

        for _ in 0..10 {
            let fake_condition_checker = FakeConditionChecker {};
            let query = random_vector(&mut rng, DIM);
            let raw_scorer = vector_holder.get_raw_scorer(query);
            let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
            graph_layers.search(TOP, EF, &scorer);
        }
    */

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
