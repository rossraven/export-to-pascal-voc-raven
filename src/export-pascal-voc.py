import os
import cv2
import numpy as np
import lxml.etree as ET
import supervisely_lib as sly
from PIL import Image
from shutil import copyfile
from collections import OrderedDict
from supervisely_lib.imaging.color import generate_rgb


my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

ARCHIVE_NAME_ENDING = '_pascal_voc.tar.gz'
RESULT_DIR_NAME_ENDING = '_pascal_voc'
RESULT_SUBDIR_NAME = 'VOCdevkit/VOC'

images_dir_name = 'JPEGImages'
ann_dir_name = 'Annotations'
ann_class_dir_name = 'SegmentationClass'
ann_obj_dir_name = 'SegmentationObject'

trainval_sets_dir_name = 'ImageSets'
trainval_sets_main_name = 'Main'
trainval_sets_segm_name = 'Segmentation'

train_txt_name = 'train.txt'
val_txt_name = 'val.txt'

pascal_contour_thickness = 7
pascal_contour_color = [224, 224, 192]
pascal_ann_ext = '.png'
pascal_contour_name = 'pascal_contour'

train_split_coef = 0.8

TRAIN_TAG_NAME = 'train'
VAL_TAG_NAME = 'val'
SPLIT_TAGS = set([TRAIN_TAG_NAME, VAL_TAG_NAME])

VALID_IMG_EXT = set(['.jpe', '.jpeg', '.jpg'])
SUPPORTED_GEOMETRY_TYPES = set([sly.Bitmap, sly.Polygon])

if train_split_coef > 1 or train_split_coef < 0:
    raise ValueError('train_val_split_coef should be between 0 and 1, your data is {}'.format(train_split_coef))


def from_ann_to_instance_mask(ann, mask_outpath):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, pascal_contour_color)
            continue

        label.geometry.draw_contour(mask, pascal_contour_color, pascal_contour_thickness)
        label.geometry.draw(mask, label.obj_class.color)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(mask_outpath, mask)


def from_ann_to_class_mask(ann, mask_outpath):
    exist_colors = [[0, 0, 0], pascal_contour_color]
    need_colors = len(ann.labels) - len(exist_colors) + 2
    for _ in range(need_colors):
        mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
        for label in ann.labels:
            if label.obj_class.name == "neutral":
                label.geometry.draw(mask, pascal_contour_color)
                continue

            new_color = generate_rgb(exist_colors)
            exist_colors.append(new_color)
            label.geometry.draw_contour(mask, pascal_contour_color, pascal_contour_thickness)
            label.geometry.draw(mask, new_color)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(mask_outpath, mask)


def ann_to_xml(project_info, image_info, img_filename, result_ann_dir, ann):
    xml_root = ET.Element("annotation")

    ET.SubElement(xml_root, "folder").text = "VOC_" + project_info.name
    ET.SubElement(xml_root, "filename").text = img_filename

    xml_root_source = ET.SubElement(xml_root, "source")
    ET.SubElement(xml_root_source, "database").text = "Supervisely Project ID:" + str(project_info.id)
    ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
    ET.SubElement(xml_root_source, "image").text = "Supervisely Image ID:" + str(image_info.id)

    xml_root_size = ET.SubElement(xml_root, "size")
    ET.SubElement(xml_root_size, "width").text = str(image_info.width)
    ET.SubElement(xml_root_size, "height").text = str(image_info.height)
    ET.SubElement(xml_root_size, "depth").text = "3"

    ET.SubElement(xml_root, "segmented").text = "1" if len(ann.labels) > 0 else "0"

    for label in ann.labels:
        if label.obj_class.name == "neutral":
            continue

        bitmap_to_bbox = label.geometry.to_bbox()

        xml_ann_obj = ET.SubElement(xml_root, "object")
        ET.SubElement(xml_ann_obj, "name").text = label.obj_class.name
        ET.SubElement(xml_ann_obj, "pose").text = "Unspecified"
        ET.SubElement(xml_ann_obj, "truncated").text = "0"
        ET.SubElement(xml_ann_obj, "difficult").text = "0"

        xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
        ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
        ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.top)
        ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
        ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.bottom)

    tree = ET.ElementTree(xml_root)

    img_name = os.path.join(result_ann_dir, os.path.splitext(img_filename)[0] + ".xml")
    ann_path = (os.path.join(result_ann_dir, img_name))
    ET.indent(tree, space="    ")
    tree.write(ann_path, pretty_print=True)


def find_first_tag(img_tags, split_tags):
    for tag in split_tags:
        if img_tags.has_key(tag):
            return img_tags.get(tag)
    return None


def write_main_set(images_stats, meta_json, result_imgsets_dir):
    result_imgsets_main_subdir = os.path.join(result_imgsets_dir, trainval_sets_main_name)
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_main_subdir)


    res_files = ["trainval.txt", "train.txt", "val.txt"]
    for file in os.listdir(result_imgsets_segm_subdir):
        if file in res_files:
           copyfile(os.path.join(result_imgsets_segm_subdir, file), os.path.join(result_imgsets_main_subdir, file))

    train_imgs = [i for i in images_stats if i['dataset'] == TRAIN_TAG_NAME]
    val_imgs = [i for i in images_stats if i['dataset'] == VAL_TAG_NAME]

    write_objs = [
        {'suffix': 'trainval', 'imgs': images_stats},
        {'suffix': 'train', 'imgs': train_imgs},
        {'suffix': 'val', 'imgs': val_imgs},
    ]
    for obj_cls in meta_json.obj_classes:
        if obj_cls.name == 'neutral':
            continue
        for o in write_objs:
            with open(os.path.join(result_imgsets_main_subdir, f'{obj_cls.name}_{o["suffix"]}.txt'), 'w') as f:
                for img_stats in o['imgs']:
                    v = "1" if obj_cls.name in img_stats['classes'] else "-1"
                    f.write(f'{img_stats["name"]} {v}\n')


def write_segm_set(images_stats, result_imgsets_dir):
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_segm_subdir)

    with open(os.path.join(result_imgsets_segm_subdir, 'trainval.txt'), 'w') as f:
        f.writelines(i['name'] + '\n' for i in images_stats)
    with open(os.path.join(result_imgsets_segm_subdir, 'train.txt'), 'w') as f:
        f.writelines(i['name'] + '\n' for i in images_stats if i['dataset'] == TRAIN_TAG_NAME)
    with open(os.path.join(result_imgsets_segm_subdir, 'val.txt'), 'w') as f:
        f.writelines(i['name'] + '\n' for i in images_stats if i['dataset'] == VAL_TAG_NAME)


@my_app.callback("from_sly_to_pascal")
@sly.timeit
def from_sly_to_pascal(api: sly.Api, task_id, context, state, app_logger):
    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(PROJECT_ID)
    meta = sly.ProjectMeta.from_json(meta_json)
    app_logger.info("Create palette")

    full_archive_name = str(project_info.id) + '_' + project_info.name + ARCHIVE_NAME_ENDING
    full_result_dir_name = str(project_info.id) + '_' + project_info.name + RESULT_DIR_NAME_ENDING

    RESULT_ARCHIVE = os.path.join(my_app.data_dir, full_archive_name)
    RESULT_DIR = os.path.join(my_app.data_dir, full_result_dir_name)
    RESULT_SUBDIR = os.path.join(RESULT_DIR, RESULT_SUBDIR_NAME)

    result_ann_dir = os.path.join(RESULT_SUBDIR, ann_dir_name)
    result_images_dir = os.path.join(RESULT_SUBDIR, images_dir_name)
    result_class_dir_name = os.path.join(RESULT_SUBDIR, ann_class_dir_name)
    result_obj_dir = os.path.join(RESULT_SUBDIR, ann_obj_dir_name)
    result_imgsets_dir = os.path.join(RESULT_SUBDIR, trainval_sets_dir_name)

    sly.fs.mkdir(result_ann_dir)
    sly.fs.mkdir(result_imgsets_dir)
    sly.fs.mkdir(result_images_dir)
    sly.fs.mkdir(result_class_dir_name)
    sly.fs.mkdir(result_obj_dir)

    app_logger.info("Pascal VOC directories has been created")

    images_stats = []
    custom_classes_colors = {}

    datasets = api.dataset.get_list(PROJECT_ID)
    for dataset in datasets:
        progress = sly.Progress('Converting images and annotations from {} dataset'.format(dataset.name), len(datasets),
                                app_logger)

        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_paths = [os.path.join(result_images_dir, image_info.name) for image_info in batch]

            api.image.download_paths(dataset.id, image_ids, image_paths)
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            for image_info, ann_info in zip(batch, ann_infos):
                img_title, img_ext = os.path.splitext(image_info.name)
                cur_img_filename = image_info.name

                cur_img_stats = {'classes': set(), 'dataset': None, 'name': img_title}
                images_stats.append(cur_img_stats)

                if img_ext not in VALID_IMG_EXT:
                    orig_image_path = os.path.join(result_images_dir, cur_img_filename)

                    jpg_image = img_title + ".jpg"
                    jpg_image_path = os.path.join(result_images_dir, jpg_image)

                    im = Image.open(orig_image_path)
                    rgb_im = im.convert("RGB")
                    rgb_im.save(os.path.join(result_images_dir, jpg_image_path))
                    os.remove(orig_image_path)

                app_logger.info(f"Images have been converted from '{img_ext}' to '.jpg'")

                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                tag = find_first_tag(ann.img_tags, SPLIT_TAGS)
                if tag is not None:
                    cur_img_stats['dataset'] = tag.meta.name

                valid_labels = []
                for label in ann.labels:
                    if type(label.geometry) in SUPPORTED_GEOMETRY_TYPES:
                        valid_labels.append(label)
                    else:
                        app_logger.warn(
                            f"Label has unsupported geometry type ({type(label.geometry)}) and will be skipped.")

                ann = ann.clone(labels=valid_labels)
                ann_to_xml(project_info, image_info, cur_img_filename, result_ann_dir, ann)
                for label in ann.labels:
                    cur_img_stats['classes'].add(label.obj_class.name)
                    custom_classes_colors[label.obj_class.name] = tuple(label.obj_class.color)

                from_ann_to_instance_mask(ann, os.path.join(result_class_dir_name, img_title + pascal_ann_ext))
                from_ann_to_class_mask(ann, os.path.join(result_obj_dir, img_title + pascal_ann_ext))

        progress.iter_done_report()

    custom_classes_colors = OrderedDict((sorted(custom_classes_colors.items(), key=lambda t: t[0])))

    with open(os.path.join(RESULT_SUBDIR, "classes_colors.txt"), "w") as cc:
         cc.write("{\n")
         for k in custom_classes_colors.keys():
             if k != list(custom_classes_colors.keys())[-1]:
                cc.write('    ' + f"'{k}': {custom_classes_colors[k]},\n")
             else:
                cc.write('    ' + f"'{k}': {custom_classes_colors[k]}\n")
         cc.write("}")

    imgs_to_split = [i for i in images_stats if i['dataset'] is None]
    train_len = int(len(imgs_to_split) * train_split_coef)

    for img_stat in imgs_to_split[:train_len]: img_stat['dataset'] = TRAIN_TAG_NAME
    for img_stat in imgs_to_split[train_len:]: img_stat['dataset'] = VAL_TAG_NAME

    write_segm_set(images_stats, result_imgsets_dir)
    write_main_set(images_stats, meta, result_imgsets_dir)

    sly.fs.archive_directory(RESULT_DIR, RESULT_ARCHIVE)
    app_logger.info("Result directory is archived")

    upload_progress = []
    remote_archive_path = "/pascal_voc_format/{}/{}".format(task_id, full_archive_name)

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(full_archive_name),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(TEAM_ID, RESULT_ARCHIVE, remote_archive_path,
                                lambda m: _print_progress(m, upload_progress))
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.full_storage_url))
    api.task.set_output_archive(task_id, file_info.id, full_archive_name, file_url=file_info.full_storage_url)

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "PROJECT_ID": PROJECT_ID
    })

    # Run application service
    my_app.run(initial_events=[{"command": "from_sly_to_pascal"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)