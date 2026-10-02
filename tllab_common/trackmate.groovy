#@ String file_in
#@ String file_out
#@ Integer MIN_AREA
#@ Integer TARGET_CHANNEL
#@ Boolean SIMPLIFY_CONTOURS
#@ Integer MAX_FRAME_GAP
#@ Double ALTERNATIVE_LINKING_COST_FACTOR
#@ Double LINKING_MAX_DISTANCE
#@ Double GAP_CLOSING_MAX_DISTANCE
#@ Double SPLITTING_MAX_DISTANCE
#@ Boolean ALLOW_GAP_CLOSING
#@ Boolean ALLOW_TRACK_SPLITTING
#@ Boolean ALLOW_TRACK_MERGING
#@ Double MERGING_MAX_DISTANCE
#@ Double CUTOFF_PERCENTILE

// Groovy port of trackmate.jy, run by trackmate_fiji2 as a separate process:
//
//     fiji --run trackmate.groovy 'MIN_AREA=20,TARGET_CHANNEL=1,...'
//
// see https://imagej.net/scripting/headless and https://imagej.net/plugins/trackmate/scripting
//
// The file paths are not passed as script parameters because the parameter syntax of --run
// (a comma separated list of key=value pairs) cannot represent Windows paths: backslashes are
// escape characters and commas / equals signs separate the parameters. They are passed in the
// environment instead (see below); the file_in and file_out parameters are only used when this
// script is run from the Fiji GUI.

import fiji.plugin.trackmate.Model
import fiji.plugin.trackmate.Settings
import fiji.plugin.trackmate.TrackMate
import fiji.plugin.trackmate.detection.LabelImageDetectorFactory
import fiji.plugin.trackmate.features.FeatureFilter
import fiji.plugin.trackmate.io.TmXmlWriter
import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory
import ij.IJ

def stdout = System.out
def log = new ByteArrayOutputStream()
try {
    file_in = System.getenv('TLLAB_TRACKMATE_FILE_IN') ?: file_in
    file_out = System.getenv('TLLAB_TRACKMATE_FILE_OUT') ?: file_out
    if (!file_in || !file_out)
        throw new IllegalArgumentException(
            'file_in and file_out must be given as the TLLAB_TRACKMATE_FILE_IN and ' +
            'TLLAB_TRACKMATE_FILE_OUT environment variables or as script parameters')

    // TrackMate logs to System.out with unicode characters, which breaks the default encoding of
    // some JVMs: collect the output and only print it when something goes wrong.
    System.setOut(new PrintStream(log, true, 'UTF-8'))

    // Get currently selected image
    def imp = IJ.openImage(file_in)
    if (imp == null)
        throw new IllegalArgumentException('Cannot read ' + file_in)

    // ----------------------------
    // Create the model object now
    // ----------------------------

    // Some of the parameters we configure below need to have
    // a reference to the model at creation. So we create an
    // empty model now.
    def model = new Model()

    // ------------------------
    // Prepare settings object
    // ------------------------

    def settings = new Settings(imp)

    // Configure detector - We use the Strings for the keys
    settings.detectorFactory = new LabelImageDetectorFactory()
    settings.detectorSettings = [
        'TARGET_CHANNEL': TARGET_CHANNEL,
        'SIMPLIFY_CONTOURS': SIMPLIFY_CONTOURS,
    ]

    // filter out spots smaller than MIN_AREA
    settings.addSpotFilter(new FeatureFilter('AREA', MIN_AREA, true))

    // Configure tracker
    settings.trackerFactory = new SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings() // almost good enough
    settings.trackerSettings['MAX_FRAME_GAP'] = MAX_FRAME_GAP
    settings.trackerSettings['ALTERNATIVE_LINKING_COST_FACTOR'] = ALTERNATIVE_LINKING_COST_FACTOR
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = LINKING_MAX_DISTANCE
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = GAP_CLOSING_MAX_DISTANCE
    settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = SPLITTING_MAX_DISTANCE
    settings.trackerSettings['ALLOW_GAP_CLOSING'] = ALLOW_GAP_CLOSING
    settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = ALLOW_TRACK_SPLITTING
    settings.trackerSettings['ALLOW_TRACK_MERGING'] = ALLOW_TRACK_MERGING
    settings.trackerSettings['MERGING_MAX_DISTANCE'] = MERGING_MAX_DISTANCE
    settings.trackerSettings['CUTOFF_PERCENTILE'] = CUTOFF_PERCENTILE

    // Add ALL the feature analyzers known to TrackMate. They will
    // yield numerical features for the results, such as speed, mean intensity etc.
    settings.addAllAnalyzers()

    // -------------------
    // Instantiate plugin
    // -------------------

    def trackmate = new TrackMate(model, settings)

    // --------
    // Process
    // --------

    if (!trackmate.checkInput())
        throw new RuntimeException(trackmate.getErrorMessage())

    if (!trackmate.process())
        throw new RuntimeException(trackmate.getErrorMessage())

    // ----------------
    // Save results
    // ----------------

    def out = new File(file_out)
    out.getParentFile()?.mkdirs()
    def writer = new TmXmlWriter(out, model.getLogger())
    writer.appendModel(model)
    writer.appendSettings(settings)
    writer.writeToFile()

    if (!out.isFile())
        throw new RuntimeException('TrackMate did not write ' + file_out)
} catch (Throwable e) {
    System.setOut(stdout)
    stdout.println(log.toString('UTF-8'))
    stdout.println('TrackMate failed: ' + e)
    e.printStackTrace(stdout)
    stdout.println('TRACKMATE_FAILED')
    throw e
} finally {
    System.setOut(stdout)
}
stdout.println('TRACKMATE_SUCCESS')