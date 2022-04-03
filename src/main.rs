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

use rand::{thread_rng, Rng};
use segment::fixtures::index_fixtures::{
    random_vector, FakeConditionChecker, TestRawScorerProducer,
};
use segment::index::hnsw_index::graph_layers::GraphLayers;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::spaces::simple::{CosineMetric, DotProductMetric, EuclidMetric};

use bit_vec::BitVec;
use segment::spaces::metric::Metric;
use segment::types::{Filter, PointOffsetType, VectorElementType};
use segment::vector_storage::simple_vector_storage::SimpleRawScorer;
use std::time::{Duration, Instant};

const M: usize = 4;
const EF_CONSTRUCT: usize = 64;
const USE_HEURISTIC: bool = true;

fn get_points() -> (Vec<Vec<VectorElementType>>, Vec<usize>) {
    (
        vec![
            vec![0.851758, 0.909671],
            vec![0.823431, 0.372063],
            vec![0.97826, 0.933157],
            vec![0.39557, 0.306488],
            vec![0.230606, 0.634397],
            vec![0.514009, 0.399594],
            vec![0.354438, 0.762611],
            vec![0.0516154, 0.733427],
            vec![0.769864, 0.288072],
            vec![0.696896, 0.509403],
            vec![0.805918, 0.923242],
            vec![0.36507, 0.513271],
            vec![0.759294, 0.128909],
            vec![0.547961, 0.877969],
            vec![0.481519, 0.909654],
            vec![0.905498, 0.0285553],
            vec![0.452462, 0.346844],
            vec![0.826136, 0.84315],
            vec![0.350968, 0.92784],
            vec![0.309734, 0.531955],
            vec![0.2932, 0.186106],
            vec![0.754536, 0.228132],
            vec![0.151885, 0.107752],
            vec![0.0787534, 0.433617],
            vec![0.347133, 0.368639],
        ],
        vec![
            0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2,
        ],
    )
}

fn random_data() -> (Vec<Vec<VectorElementType>>, Vec<usize>) {
    let count: usize = 5_000;
    let dim: usize = 128;
    let mut rng = thread_rng();

    let mut points: Vec<Vec<VectorElementType>> = vec![];
    for _ in 0..count {
        points.push(random_vector(&mut rng, dim));
    }

    (points, vec![])
}

fn main() -> std::io::Result<()> {
    let start = Instant::now();

    let mut rng = thread_rng();

    let (points, point_levels) = random_data();
    // let (points, point_levels) = get_points();

    let mut graph_layers =
        GraphLayers::new(points.len(), M, M * 2, EF_CONSTRUCT, 10, USE_HEURISTIC);
    let fake_condition_checker = FakeConditionChecker {};
    let deleted = BitVec::from_elem(points.len(), false);
    let metric = DotProductMetric {};
    for idx in 0..points.len() {
        let added_vector = points[idx].clone();
        let raw_scorer = SimpleRawScorer {
            query: metric.preprocess(&added_vector).unwrap_or(added_vector),
            metric: &metric,
            vectors: &points,
            deleted: &deleted,
        };
        let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);

        let level = if idx < point_levels.len() {
            point_levels[idx]
        } else {
            graph_layers.get_random_layer(&mut rng)
        };

        //println!("");
        //println!("Insert point {} {} {}", idx, raw_scorer.query[0], raw_scorer.query[1]);
        graph_layers.link_new_point(idx as PointOffsetType, level, &scorer);
    }
    //graph_layers.dump();

    let query = random_vector(&mut rng, points[0].len());

    let scorer = SimpleRawScorer {
        query: metric.preprocess(&query).unwrap_or(query.clone()),
        metric: &metric,
        vectors: &points,
        deleted: &deleted,
    };
    let scorer = FilteredScorer::new(&scorer, &fake_condition_checker, None);
    let found = graph_layers.search(5, EF_CONSTRUCT, &scorer);
    for idx in found.iter().copied() {
        println!("found {} with dist {}", idx.idx, idx.score);
    }

    let mut best = 0;
    let mut dist = 1000000000.;
    for idx in 0..points.len() {
        let d = metric.similarity(&query, &points[idx]);
        if d < dist {
            best = idx;
            dist = d;
        }
    }
    println!("best {} with dist {}", best, dist);

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
